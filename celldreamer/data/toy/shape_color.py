import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class ShapeColorDataset(Dataset):
    def __init__(self, num_samples):
        """
        ShapeColorDataset constructor.

        Args:
            num_samples (int): Number of samples to generate.
        """
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        Generate a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Tuple containing the image and labels.
        """
        # Set random seed
        # Generate random shape, color, size, and position
        shapes = ["circle", "square"]
        colors = ["lightblue", "lightgreen", "lightyellow", "lightgray"]
        sizes = ["small", "medium", "large"]
        positions = ["topleft", "topright", "bottomleft", "bottomright"]

        shape = random.choice(shapes)
        color = random.choice(colors)
        color_rgb = (0.13, 0.68, 0.95) if color == "lightblue" else (0.47, 0.87, 0.47) if color == "lightgreen" else (1, 1, 0.4) if color == "lightyellow" else (0.75, 0.75, 0.75) if color == "lightgray" else (0, 0, 0)
        size = random.choice(sizes)
        position = random.choice(positions)

        # Create a simple image with shape, color, size, and position
        image = torch.ones(3, 32, 32)  # Assuming RGB images of size 32x32

        if shape == "circle":
            radius = random.randint(4, 8) if size == "small" else random.randint(8, 12) if size == "medium" else random.randint(12, 16)
            center_x = 8 if "top" in position else 24
            center_y = 8 if "left" in position else 24

            for i in range(32):
                for j in range(32):
                    if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
                        # fill image with color
                        image[0, i, j] = color_rgb[0]
                        image[1, i, j] = color_rgb[1]
                        image[2, i, j] = color_rgb[2]

        elif shape == "square":
            len_x = 4 if size == "small" else 8 if size == "medium" else 12
            len_y = 4 if size == "small" else 8 if size == "medium" else 12

            start_x = 4 if "top" in position else 32-len_x-4
            start_y = 4 if "left" in position else 32-len_y-4


            end_x = start_x + len_x
            end_y = start_y + len_y

            image[0, start_x:end_x+1, start_y:end_y+1] = color_rgb[0]
            image[1, start_x:end_x+1, start_y:end_y+1] = color_rgb[1]
            image[2, start_x:end_x+1, start_y:end_y+1] = color_rgb[2]

            # does not consider position
            # start_x = 7 if size == "small" else 5 if size == "medium" else 3
            # start_y = 7 if size == "small" else 5 if size == "medium" else 3
            # end_x = 25 if size == "small" else 27 if size == "medium" else 29
            # end_y = 25 if size == "small" else 27 if size == "medium" else 29

            # image[0, start_x:end_x+1, start_y:end_y+1] = color_rgb[0]
            # image[1, start_x:end_x+1, start_y:end_y+1] = color_rgb[1]
            # image[2, start_x:end_x+1, start_y:end_y+1] = color_rgb[2]

        # Apply transformations
        image = self.transform(image)

        # Create condition labels
        shape_label = shapes.index(shape)
        color_label = colors.index(color)
        size_label = sizes.index(size)
        position_label = positions.index(position)

        batch = {
            "X": image,
            "y": {"y_shapes": shape_label,
                  "y_colors": color_label,
                  "y_sizes": size_label,
                  "y_positions": position_label}
        }

        return batch