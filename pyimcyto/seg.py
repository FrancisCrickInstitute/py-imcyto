import math
import os
from pickle import load

import numpy as np
import pandas as pd
import skimage.io as io
import tensorflow as tf
from pyimcyto.losses import make_loss
from pyimcyto.models import error_model, nested_unet
from pyimcyto.postprocessing import instance_closing, probability_basin_watershed_2
from pyimcyto.util import stitch_with_overlap, tilegen, trainGenerator
from skimage.measure import regionprops_table
from skimage.morphology import diamond
from skimage.segmentation import expand_labels, find_boundaries, relabel_sequential
from skimage.util import img_as_float32, img_as_uint, map_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class deepimcyto:
    def __init__(self, weights_path):
        """The deepimcyto segmentation class.

        Args:
            weights_path (str): Path to the directory containing the model weights.
        """

        # POSTPROCESSING WATERSHED THRESHOLDS:
        self.NUCLEUS_CONFIDENCE = 0.5
        self.COMS_CONFIDENCE = 0.5
        self.COMS_CONFIDENCE_LOW = (
            0.125  # lower confidence for re-segmenting low likelihood regions
        )
        self.THRESH_NUC = self.NUCLEUS_CONFIDENCE * 255
        self.THRESH_COM = self.COMS_CONFIDENCE * 255
        self.THRESH_COM_LOW = self.COMS_CONFIDENCE_LOW * 255
        self.MIN_OBJ_SIZE = 4
        self.WATERSHED_LINE = False
        self.COMPACTNESS = 0
        self.gpus = len(tf.config.list_physical_devices("GPU"))
        self.nuc_weights = os.path.join(weights_path, "nucleus_edge_weighted.hdf5")
        self.com_weights = os.path.join(weights_path, "com.hdf5")
        self.AE_weights = os.path.join(weights_path, "AE_weights.hdf5")
        self.boundary_weights = os.path.join(weights_path, "boundaries.hdf5")
        self.scaler_path = os.path.join(weights_path, "nuclear_morph_scaler.pkl")
        self.morph_scaler = load(open(self.scaler_path, "rb"))
        self.input_shape = (512, 512, 1)
        self.nuc_model = self.create_model(
            self.nuc_weights, self.input_shape, self.gpus
        )
        self.bound_model = self.create_model(
            self.boundary_weights, self.input_shape, self.gpus
        )
        self.com_model = self.create_model(
            self.com_weights, self.input_shape, self.gpus
        )
        self.thresh_nuc = 0.5
        self.thresh_com = 0.5
        self.thresh_com_low = 0.125
        self.randomise = True
        self.test_images = []
        self.prediction_masks = []
        self.prediction_boundaries = []
        self.prediction_nuclei = []
        self.prediction_coms = []
        self.error_images = []
        self.dilation_radius = 5
        self.aug_dict = dict(
            rotation_range=0.2,  # default augmentation parameters for training
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        self.train_batch_size = 4
        self.custom_model_name = "model"
        self.checkpoint_dir = "./checkpoints"
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.custom_model_name}_checkpoint.hdf5"
        )
        self.train_steps_per_epoch = 200
        self.train_epochs = (200,)
        self.val_steps_per_epoch = None

        # features for autoencoder:
        self.features = [
            "area",
            "eccentricity",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "moments_hu-0",
            "moments_hu-1",
            "moments_hu-2",
            "moments_hu-3",
            "moments_hu-4",
            "moments_hu-5",
            "moments_hu-6",
            "perimeter",
            "perimeter_crofton",
            "solidity",
        ]

        self.error_model = error_model(self.features, self.AE_weights)
        self.perform_instance_closing = True

        print("Initialised deep-imcyto model.")

    def summary(self):
        """Show summary of deep-imcyto model."""
        print("\nNucleus Model:")
        self.nuc_model.summary()
        print("\nBoundary Model:")
        self.bound_model.summary()
        print("\nCentre of mass Model:")
        self.com_model.summary()
        print("\nAutoencoder error Model:")
        self.error_model.summary()

    def display(self):
        """Display Configuration values."""
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("\n~~~~ deep-imcyto config ~~~~")
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def predict(
        self,
        input,
        tile_shape=(512, 512, 1),
        overlap=(1, 1),
        min_size=5,
        error_thresh=1,
    ):
        """Predict instance segmentation masks for a given input image."""

        self.test_images.append(input)
        nuc = self.predict_tiles(self.nuc_model, input, tile_shape, overlap)
        boundary = self.predict_tiles(self.bound_model, input, tile_shape, overlap)
        com = self.predict_tiles(self.com_model, input, tile_shape, overlap)

        label = probability_basin_watershed_2(
            nuc,
            boundary,
            com,
            thresh_nuc=self.thresh_nuc,
            thresh_com=self.thresh_com,
            min_obj_size=min_size,
            compactness=0,
        )

        if self.perform_instance_closing and self.gpus == True:
            # Perform instance closing:
            selem = diamond(1)
            print("Performing instance closing...")
            label = instance_closing(label, strel=selem)
            print("Done.")

        # Perform anomaly detection:
        label, error_img = self.process_anomalies(label)
        self.error_images.append(error_img)

        # if any elements of the error array are greater than the error cutoff (==1), reprocess the regions that have this large error, else use this refinec mask as the final one:
        if np.any(error_img > error_thresh):
            label = self.reprocess_unlikely_labels(
                label,
                error_img,
                boundary,
                nuc,
                com,
                self.thresh_com_low,
                self.thresh_nuc,
                self.randomise,
            )
        else:
            label = self.randomise_labels(label) if self.randomise else label

        self.prediction_masks.append(label)
        self.prediction_boundaries.append(boundary)
        self.prediction_nuclei.append(nuc)
        self.prediction_coms.append(com)

        return label

    def train_custom(self, train_path, img_dir, mask_dir, val_path, initial_weights):
        """Train a custom model on a given dataset.

        NOTE: This is a work in progress.
        """
        self.loss = make_loss("bce_dice")

        custom_model = self.create_model(initial_weights, self.input_shape, self.gpus)

        train_gen = trainGenerator(
            self.train_batch_size, train_path, img_dir, mask_dir, self.aug_dict
        )
        val_gen = trainGenerator(
            self.train_batch_size, val_path, img_dir, mask_dir, self.aug_dict
        )
        model_checkpoint = ModelCheckpoint(
            self.checkpoint_path, monitor="val_loss", verbose=1, save_best_only=True
        )
        custom_model.fit_generator(
            train_gen,
            steps_per_epoch=self.train_steps_per_epoch,
            epochs=self.train_epochs,
            callbacks=[model_checkpoint],
            validation_data=val_gen,
            validation_steps=self.val_steps_per_epoch,
        )

        return self.custom_model

    def get_boundaries(self):
        """Get boundaries from prediction masks."""
        self.prediction_boundaries = [
            find_boundaries(x, mode="outer") for x in self.prediction_masks
        ]
        return self.prediction_boundaries

    def dilate_nuclei(self, radius=5):
        self.dilated_masks = [expand_labels(x, radius) for x in self.prediction_masks]
        return self.dilated_masks

    def make_overlays(self, alpha=0.5):
        """Make an overlay of the input image and the prediction mask."""
        self.overlays = [
            self.overlay(i, m, alpha)
            for i, m in zip(self.test_images, self.prediction_boundaries)
        ]

    def overlay(self, image, mask, alpha=0.5):
        """Overlay a binary mask on an input image.
        
        Args:
            image (np.array): Input image.
            mask (np.array): Binary mask.
            alpha (float, optional): Alpha value for the overlay. Defaults to 0.5.

        Returns:
            np.array: Overlayed image.
        """
        image = image / np.max(image)
        mask = mask / np.max(mask)
        image = img_as_float32(image)
        mask = img_as_float32(mask)
        image = np.stack([image, image, image], axis=-1)
        mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        return image + mask * alpha

    def predict_tiles(self, model, image, tile_shape, overlap):
        """Create a tile generator for a large image and predict tiles for a given model."""

        testGene = tilegen(image, overlap=overlap)
        if self.gpus:
            with tf.device("/GPU:0"):
                results = model.predict(testGene, 1, verbose=1)
        else:
            with tf.device("/CPU:0"):
                results = model.predict(testGene, 1, verbose=1)
        stitched = stitch_with_overlap(
            results, image.shape, tile_shape=tile_shape, overlap=overlap
        )
        return stitched

    def create_model(self, weights_path, tile_shape, gpus):
        """Create a U-net++ model with given weights and input shape."""
        if gpus:
            with tf.device("/GPU:0"):
                model = nested_unet(
                    pretrained_weights=weights_path, input_size=tile_shape
                )
        else:
            with tf.device("/CPU:0"):
                model = nested_unet(
                    pretrained_weights=weights_path, input_size=tile_shape
                )
        return model

    def anomaly_detect(
        self, mask, model, model_features, region_properties, scaler, batch_size=256
    ):
        """
        Detect morphologically suspect nuclei from simple morph features.
        """

        test_measure = regionprops_table(mask, properties=region_properties)
        test_measure_df = pd.DataFrame(test_measure)

        # predict on data
        if self.gpus:
            with tf.device("/GPU:0"):
                # model.predict(test_measure_df[model_features].values)
                X_scaled = scaler.transform(test_measure_df[model_features].values)
                results = model.predict(X_scaled, batch_size=batch_size)
        else:
            with tf.device("/CPU:0"):
                # model.predict(test_measure_df[model_features].values)
                X_scaled = scaler.transform(test_measure_df[model_features].values)
                results = model.predict(X_scaled, batch_size=batch_size)

        # get errors:
        reconstruction_errors = tf.keras.losses.msle(results, X_scaled)

        if results.shape[0] != 1:
            results = np.squeeze(results)

        # create dataframe with prediction errors:
        test_measure_df = test_measure_df.reset_index()
        test_measure_df.loc[:, "autoencoder_prediction_MSE"] = pd.Series(
            reconstruction_errors.numpy()
        )
        return test_measure_df

    def process_anomalies(
        self, mask, save_error_image=False, outdir=None, imagename=None
    ):
        """Measure morphology and find anomalous nuclei relative to ground truth morphologies.

        Args:
            mask (np.array): Input label image.
            save_error_image (bool, optional): Whether to save an image of the errors. Defaults to False.
            outdir (str, optional): Output directory for error image. Defaults to None.
            imagename (str, optional): Name of image. Defaults to None.

        Returns:
            tuple(mask_refined, error_img): Refined mask with anomalous labels removed, and image of errors for each label.
            mask_refined (np.array): Refined mask with anomalous labels removed.
            error_img (np.array): Image of errors for each label.
        """

        # properties to measure per label
        test_properties = [
            "centroid",
            "label",
            "area",
            "eccentricity",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "moments_hu",
            "perimeter",
            "perimeter_crofton",
            "solidity",
        ]

        self.morphology = self.anomaly_detect(
            mask, self.error_model, self.features, test_properties, self.morph_scaler
        )
        mask_refined = self.remove_anomalous_labels(self.morphology, mask)
        error_img = map_array(
            mask,
            self.morphology["label"].values,
            self.morphology["autoencoder_prediction_MSE"].values,
        )
        error_img = img_as_float32(error_img)

        if save_error_image:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            spath = os.path.join(outdir, f"{imagename}_AE_error.tiff")
            io.imsave(spath, error_img)

        return mask_refined, error_img

    def remove_anomalous_labels(self, AE_results, mask, threshold=1):
        """Remove labels with high anomaly detection error from a label image.

        Args:
            AE_results (pd.DataFrame): Dataframe of anomaly detection results.
            mask (np.array): Input label image.
            threshold (int, optional): Threshold for anomaly detection error. Defaults to 1.

        Returns:
            np.array: Refined mask with anomalous labels removed.
        """
        high_error_df = AE_results[AE_results["autoencoder_prediction_MSE"] > threshold]
        high_error_labels = high_error_df["label"].values
        logicalmask = np.isin(mask, high_error_labels.tolist())
        modmask = np.where(logicalmask == 0, mask, 0)
        return modmask

    def reprocess_unlikely_labels(
        self,
        mask,
        error_mask,
        boundary_im,
        nuc_im,
        com_im,
        com_thresh_lower,
        thresh_nuc,
        randomise,
    ):
        """Reprocess labels with high anomaly detection error.

        Args:
            mask (np.array): Input label image.
            error_mask (np.array): Image of errors for each label.
            boundary_im (np.array): Boundary prediction image.
            nuc_im (np.array): Nucleus pprediction image.
            com_im (np.array): centre of mass prediction mage.
            com_thresh_lower (float): Threshold for centre of mass prediction in the reprocessed labels.
            thresh_nuc (float): Threshold for nucleus prediction in the reprocessed labels.
            randomise (bool): Whether to randomise the labels in the reprocessed labels.

        Returns:
            np.array: Refined mask with anomalous labels removed.
        """

        # create waterhed image with lower com limit
        watershed_im = probability_basin_watershed_2(
            nuc_im, boundary_im, com_im, thresh_nuc, com_thresh_lower
        )

        # get watershed labels with lower coms limit, restricted to high error areas of original watershed:
        high_error_iter = np.where(error_mask > 1, watershed_im, 0)

        # perform anomaly detection on this new iteration of watershed:
        (
            processed,
            error_mask_2,
        ) = self.process_anomalies(high_error_iter)

        # remove unlikely labels from the second iteration of watershed entirely:
        processed = np.where(error_mask_2 > 1, 0, processed)

        # do instance closing on resultant label image:
        diamond_selem = diamond(1)
        processed = instance_closing(processed, diamond_selem)

        # relabel so that labels are sequential integers
        relab, fw, inv = relabel_sequential(processed)

        # add the maximum label value from the good part of the original watershed mask to these labels:
        relab_add = np.where(relab > 0, relab + np.amax(mask), 0)

        # join the second iteration of watershed into the original watershed:
        joined_mask = np.where(mask == 0, relab_add, mask)

        if randomise:
            joined_mask = self.randomise_labels(joined_mask)

        return joined_mask

    def randomise_labels(self, mask):
        """Randomise the labels in a label image.

        Args:
            mask (np.array): Input label image.

        Returns:
            np.array: Resultant mask with randomised labels.
        """
        new_labels = np.arange(1, len(np.unique(mask)))
        np.random.shuffle(new_labels)
        old_labels = np.unique(mask[mask > 0])

        reordered_label_img = map_array(mask, old_labels, new_labels)
        if np.amax(reordered_label_img) <= 65535:
            reordered_label_img = img_as_uint(reordered_label_img)
        else:
            reordered_label_img = reordered_label_img / np.amax(reordered_label_img)
            reordered_label_img = img_as_float32(reordered_label_img)

        return reordered_label_img
