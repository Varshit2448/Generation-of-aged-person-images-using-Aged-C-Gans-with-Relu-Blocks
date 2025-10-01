🧠 Generation-of-aged-person-images-using-Aged-C-Gans-with-Relu-Blocks

This project trains a lightweight GAN to generate aged face images using the UTKFace dataset. It includes:

✅ Data loading & preprocessing
✅ Encoder, Generator, Discriminator, Age Embedding networks
✅ Custom training loop
✅ Model checkpointing
✅ Inference function to age a given face image

✅ 1. Requirements

Make sure you have:

Python 3.x

TensorFlow 2.x

UTKFace dataset (JPG format)

✅ 2. Dataset Setup

Place the UTKFace dataset in one of these folders:

/kaggle/input/utkface-new/UTKFace
/kaggle/input/utkface/UTKFace
./UTKFace


Each file should follow:

{age}_{gender}_{race}_{date}.jpg

Example input image:


<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/14dea59c-a9ea-47b0-8db2-500200ca87a8" />




✅ 3. Training

Run the script directly:

python your_script.py


The script will:

✔ Find the UTKFace folder
✔ Build the dataset
✔ Train for epochs = 50
✔ Save models after every epoch in:

checkpoints_utk/


You will see logs like:

Epoch 1 Step 50 | D_loss: ... G_loss: ...
Starting epoch 2/50
...
Training finished.

✅ 4. Model Components
🔹 Encoder (E)

Extracts a latent vector from an input face.

🔹 Age Embedding (A)

Converts a normalized age (0–1) into an embedding.

🔹 Generator (G)

Takes:

Latent vector

Target age embedding
Outputs:

A synthesized face image

🔹 Discriminator (D)

Predicts:

Real vs fake

Age correctness

✅ 5. Inference (Test on a New Image)

You can generate an output face after training using:

from your_script import infer_image

output = infer_image(
    image_path="input.jpg",
    target_age=50,      # any age 0–116
    out_path="output.jpg"
)


This will:

Load and resize the input image

Predict a latent vector

Embed the target age

Generate the new face

Save/display the result

✅ 6. Key Configs (Change in Config)
Parameter	Value	Purpose
img_size	64	Output/input resolution
batch_size	2	Training batch size
latent_dim	256	Encoder/Generator latent size
age_embed_dim	64	Age embedding size
epochs	50	Total training epochs
steps_per_epoch	500	Iterations per epoch
✅ 7. Checkpoints

Models are saved as:

checkpoints_utk/
│
├── E_epochX.keras
├── A_epochX.keras
├── G_epochX.keras
└── D_epochX.keras


You can manually load them using load_models().

✅ 8. Notes & Tips

Generated images will be low-res (64×64) due to training settings

Increase epochs or image size for better quality

Use infer_image() after training completes

✅ 9. Flow Chart

<img width="1410" height="758" alt="Screenshot 2025-10-01 092756" src="https://github.com/user-attachments/assets/00cf498b-b10e-4966-80d3-c94169b25e9a" />

✅ 10. Versions

This code is Built on an iterative designing mode here you can see outputs of various versions

Version 1.0 with classic GANs

![aged_face (1)](https://github.com/user-attachments/assets/75471cd2-2ebd-4421-b720-041aa44f8c45)

Version 1.1 with Conditional Aged GANs

![aged_face1](https://github.com/user-attachments/assets/2190eff2-b74d-4579-913c-f9674f56a87c)

Version 2.1 with Conditional Aged GANs where ReLU blocks are Embedded into like LSTM networks but this version is an experiment with RELU blocks embedded into only generator model only

<img width="472" height="475" alt="Screenshot 2025-09-28 125822" src="https://github.com/user-attachments/assets/855dbe44-80b3-40c4-9a0a-aceddd0560d4" />

Versions 2.2 Conditional Aged GANs where ReLU blocks are Embedded into all four networks

<img width="293" height="295" alt="Screenshot 2025-09-27 162154" src="https://github.com/user-attachments/assets/764dd732-896f-48a1-8475-1a0e0dbbf1e4" />

✅ 11. Limitations

✅ 1. Low Image Resolution (64×64)

The model is trained and generated at only 64×64 pixels, which limits visual quality and realism. Facial details, textures, and identity preservation are weak at this scale 

✅ 2. Small Batch Size (batch_size = 2)

The very small batch size may lead to:

Slower training stability

Poor gradient estimation

Noisy discriminator/generator updates

This restriction is mainly due to memory management.

✅ 3. Simplified Generator Architecture

The generator uses basic Conv2DTranspose layers with few filters and no attention mechanisms or skip connections. This limits:

High-frequency detail generation

Identity retention

Robust age transformation

✅ 4. No Identity Preservation Loss

There is no face identity constraint like:

Feature matching

Face recognition loss

Identity embedding comparison
So, the aged output often distorts facial structure and realism.

✅ 5. Limited Age Conditioning

The model treats age as a single normalized scalar, without:

Class buckets

Explicit target-age guidance

Semantic control over age realism

This weak conditioning can cause mismatched or random transformations.

These Limitations are majorly to lack of computational resources so this project is available for further contributions with code optimisations







