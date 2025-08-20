from xml_parser import extract_text_from_xml
import tensorflow as tf


class OpenIDataset(tf.keras.utils.Sequence):
    def __init__(self, metadata_path, tokenizer, image_size=(128, 128)):
        self.metadata = pd.read_csv(metadata_path)
        self.tokenizer = tokenizer
        self.image_size = image_size

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 加载影像
        img = tf.image.decode_png(tf.io.read_file(row['image_path']), channels=1)
        img = tf.image.resize(img, self.image_size) / 255.0

        # 从XML加载文本
        text = extract_text_from_xml(row['report_path'])
        text_encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="tf"
        )

        return (img, text_encoded['input_ids'][0]), row['label']