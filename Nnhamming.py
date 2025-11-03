import numpy as np
import os
from pathlib import Path
from PIL import Image
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv
import cv2


class HammingNetworkTrafficSigns:
    """
    Implementa una red de Hamming para reconocimiento de señales de tránsito.
    Permite entrenar con imágenes de tres clases (pare, ceda, via prioritaria), clasificar nuevas imágenes,
    visualizar patrones y procesar lotes de imágenes de prueba.
    Usa umbralización Otsu para manejar variaciones de iluminación.
    """

    def __init__(self, img_size=32, num_patterns=3, dataset_dir="dataset"):
        """
        Inicializa la red de Hamming.
        Args:
            img_size (int): Tamaño de las imágenes.
            num_patterns (int): Número de clases/patrones.
            dataset_dir (str or Path): Directorio base del dataset.
        """
        self.img_size = img_size
        self.input_size = img_size * img_size
        self.weights = []
        self.labels = ["via prioritaria", "ceda", "pare"]
        self.base_dir = Path(dataset_dir).resolve()
        self.output_dir = self.base_dir / "test"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_input_path = self.output_dir / "datainput.csv"
        self.csv_output_path = self.output_dir / "dataoutput.csv"

    def _preprocess_image(self, image_path, use_otsu=True):
        """
        Carga y binariza una imagen en escala de grises con Otsu para iluminación variable.
        Args:
            image_path (str or Path): Ruta de la imagen.
            use_otsu (bool): Si True, usa Otsu; else fixed threshold.
        Returns:
            np.ndarray: Vector binarizado (-1, 1).
        """
        try:
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image file {image_path} not found.")
            image = Image.open(image_path).convert("L")
            if image.size != (self.img_size, self.img_size):
                raise ValueError(
                    f"Image must be {self.img_size}x{self.img_size} pixels."
                )

            gray = np.array(image, dtype=np.uint8)
            gray = cv2.equalizeHist(gray)

            if use_otsu:
                try:
                    _, binary = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    print(f"Otsu threshold applied successfully for {image_path}.")
                except cv2.error as cv_err:
                    print(
                        f"Otsu failed (low contrast?): {cv_err}. Falling back to fixed threshold."
                    )
                    binary = np.where(gray > 128, 255, 0).astype(np.uint8)
            else:
                binary = np.where(gray > 128, 255, 0).astype(np.uint8)

            flat = (binary / 255).flatten()
            return np.where(flat == 0, -1, 1)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            raise

    def train(self, dataset_dir, num_samples_per_class=40, use_otsu=True):
        """
        Entrena la red con imágenes de las clases definidas.
        Args:
            dataset_dir (str or Path): Directorio con subcarpetas de cada clase.
            num_samples_per_class (int): Número de imágenes por clase.
            use_otsu (bool): Si True, usa Otsu en grayscale.
        """
        dataset_dir = Path(dataset_dir).resolve()
        if not dataset_dir.exists():
            print(f"Error: Dataset directory {dataset_dir} does not exist.")
            return

        self.input_size = self.img_size * self.img_size

        print("Starting training...")
        print(f"Expected classes: {self.labels}")

        for class_name in self.labels:
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                print(f"Error: Directory {class_dir} does not exist.")
                continue

            images = [
                f
                for f in class_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".ppm")
            ]
            if not images:
                print(f"Error: No images found in {class_name}.")
                continue
            if len(images) < num_samples_per_class:
                print(
                    f"Warning: Class {class_name} has {len(images)} images, using all."
                )
                num_samples_per_class = len(images)

            print(f"Processing class: {class_name} ({len(images)} images available)")
            class_patterns = []
            for i in range(min(num_samples_per_class, len(images))):
                img_path = images[i]
                print(f"  - Loading image: {img_path}")
                try:
                    pattern = self._preprocess_image(img_path, use_otsu)
                    class_patterns.append(pattern)
                except Exception as e:
                    print(f"  - Skipping image {img_path}: {e}")
                    continue

            if class_patterns:
                avg_pattern = np.mean(class_patterns, axis=0)
                avg_pattern = np.where(avg_pattern > 0, 1, -1)
                self.weights.append(avg_pattern)
                print(f"  - Average pattern for {class_name} stored.")
            else:
                print(f"  - No valid patterns for {class_name}.")

        if len(self.weights) != 3:
            print(f"Warning: Expected 3 classes, but trained {len(self.weights)}.")
        print("Training completed.")

    def classify(self, image_path, use_otsu=True):
        """
        Clasifica una imagen de entrada.
        Args:
            image_path (str or Path): Ruta de la imagen.
            use_otsu (bool): Si True, usa Otsu en grayscale.
        Returns:
            Tuple: (etiqueta detectada, distancia de Hamming, índice de clase, flag de detección)
        """
        try:
            input_pattern = self._preprocess_image(image_path, use_otsu)
        except Exception as e:
            print(f"Error classifying {image_path}: {e}")
            return "none", float("inf"), -1, 0

        print(f"Classifying image: {image_path}")
        min_distance = float("inf")
        best_label = "none"
        best_index = -1

        for i, weight in enumerate(self.weights):
            hamming_dist = np.sum(input_pattern != weight) / 2
            print(f"  - Distance to class {self.labels[i]}: {hamming_dist:.2f}")
            if hamming_dist < min_distance:
                min_distance = hamming_dist
                best_label = self.labels[i]
                best_index = i

        pattern_detected = 1
        detected_label = best_label

        return detected_label, min_distance, best_index, pattern_detected

    def inspect_network(self):
        """
        Muestra información sobre la red y los patrones almacenados.
        """
        print("\n=== Network Inspection ===")
        print(f"Image size: {self.img_size}x{self.img_size}")
        print(f"Input vector size: {self.input_size} bits")
        print(f"Defined classes: {self.labels}")
        print(f"Number of stored patterns: {len(self.weights)}")
        if self.weights:
            print("Pattern details:")
            for i, (label, weight) in enumerate(zip(self.labels, self.weights)):
                print(f"  - Class {label}: Pattern of {weight.shape[0]} bits")
        else:
            print("  - No patterns stored (training not performed).")

    def visualize_patterns(self):
        """
        Visualiza los patrones almacenados para cada clase.
        """
        if not self.weights:
            print("Error: No patterns stored to visualize.")
            return

        fig, axes = plt.subplots(
            1, len(self.weights), figsize=(4 * len(self.weights), 4)
        )
        if len(self.weights) == 1:
            axes = [axes]

        for i, (weight, label) in enumerate(zip(self.weights, self.labels)):
            pattern = np.where(weight == 1, 255, 0).reshape(
                self.img_size, self.img_size
            )
            axes[i].imshow(pattern, cmap="gray")
            axes[i].set_title(f"Pattern: {label}")
            axes[i].axis("off")
        plt.show()

    def visualize(self, image_path, predicted_label, best_index):
        """
        Visualiza la imagen de entrada y el patrón predicho.
        Args:
            image_path (str or Path): Ruta de la imagen.
            predicted_label (str): Etiqueta predicha.
            best_index (int): Índice del patrón predicho.
        """
        try:
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image file {image_path} not found.")
            input_img = np.array(Image.open(image_path).convert("L"))
            pred_pattern = np.where(self.weights[best_index] == 1, 255, 0).reshape(
                self.img_size, self.img_size
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(input_img, cmap="gray")
            ax1.set_title("Input Image")
            ax2.imshow(pred_pattern, cmap="gray")
            ax2.set_title(f"Prediction: {predicted_label}")
            plt.show()
        except Exception as e:
            print(f"Error visualizing {image_path}: {e}")

    def process_test_images(self, test_dir, use_otsu=True):
        """
        Procesa imágenes de prueba, lee datainput.csv y genera dataoutput.csv con resultados.
        Args:
            test_dir (str or Path): Directorio de imágenes de prueba.
            use_otsu (bool): Si True, usa Otsu en grayscale.
        """
        test_dir = Path(test_dir).resolve()
        if not test_dir.exists():
            print(f"Error: Test directory {test_dir} does not exist.")
            return
        if not self.csv_input_path.exists():
            print(f"Error: File {self.csv_input_path} does not exist.")
            return

        input_data = []
        try:
            with open(self.csv_input_path, mode="r", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    if len(row) < 3:
                        print(f"Warning: Invalid row in datainput.csv: {row}")
                        continue
                    image_name, has_pattern, pattern = row[0], row[1], row[2]
                    input_data.append(
                        {
                            "image_name": image_name,
                            "has_pattern": int(has_pattern),
                            "pattern": pattern,
                        }
                    )
            print(f"Read {len(input_data)} entries from {self.csv_input_path}")
        except Exception as e:
            print(f"Error reading {self.csv_input_path}: {e}")
            return

        output_data = []
        for entry in input_data:
            image_name = entry["image_name"]
            image_path = test_dir / image_name
            if not image_path.exists():
                print(f"Error: Image {image_path} does not exist.")
                continue

            print(f"\nProcessing test image: {image_path}")
            label, min_distance, index, pattern_detected = self.classify(
                image_path, use_otsu=use_otsu
            )
            print(f"Detected signal: {label} (Hamming Distance: {min_distance:.2f})")

            output_data.append(
                [
                    image_name,
                    entry["has_pattern"],
                    entry["pattern"],
                    pattern_detected,
                    label,
                ]
            )

        try:
            with open(
                self.csv_output_path, mode="w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "image_name",
                        "has_pattern",
                        "pattern",
                        "pattern_detected",
                        "detected_pattern",
                    ]
                )
                writer.writerows(output_data)
            print(f"Results saved to: {self.csv_output_path}")
        except Exception as e:
            print(f"Error saving {self.csv_output_path}: {e}")


if __name__ == "__main__":
    """
    Example usage of HammingNetworkTrafficSigns.
    Ensure the following dependencies are installed:
    pip install numpy pillow matplotlib opencv-python
    Dataset structure:
    dataset/
    ├── pare/
    ├── ceda/
    ├── via prioritaria/
    ├── test/
    └── test/datainput.csv
    """
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset"
    test_dir = dataset_dir / "test"
    print(f"Dataset directory: {dataset_dir}")
    print(f"Test directory: {test_dir}")
    img_size = 32
    use_otsu = True

    net = HammingNetworkTrafficSigns(
        img_size=img_size, num_patterns=3, dataset_dir=dataset_dir
    )

    print("Before training:")
    net.inspect_network()

    try:
        net.train(dataset_dir, num_samples_per_class=50, use_otsu=use_otsu)
    except Exception as e:
        print(f"Training failed: {e}")

    print("\nAfter training:")
    net.inspect_network()

    print("\nVisualizing stored patterns:")
    try:
        net.visualize_patterns()
    except Exception as e:
        print(f"Visualization failed: {e}")

    try:
        net.process_test_images(test_dir, use_otsu=use_otsu)
    except Exception as e:
        print(f"Test processing failed: {e}")
