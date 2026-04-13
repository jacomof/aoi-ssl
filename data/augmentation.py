import numpy as np
import albumentations as A


class RandomResizedCropWithMask(A.RandomResizedCrop):
    def get_params_dependent_on_data(self, params, data):
        if "mask" in data:
            # Process the mask (using similar logic as in CropNonEmptyMaskIfExists)
            mask = data["mask"]
            # (Optionally include preprocessing similar to _preprocess_mask)
            mask_height, mask_width = mask.shape[:2]
            if mask.any():
                non_zero_yx = np.argwhere(mask)
                y, x = self.py_random.choice(non_zero_yx)
                x = np.clip(x, 0, mask_width - self.size[1])
                y = np.clip(y, 0, mask_height - self.size[0])
                x_max, y_max = x + self.size[1], y + self.size[0]
                crop_coords = (
                    x,
                    y,
                    x_max,
                    y_max,
                )
                return {"crop_coords": crop_coords}
        # Fallback to standard random resized crop
        return super().get_params_dependent_on_data(params, data)
