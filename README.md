🧠 Age Progression GAN – UTKFace Training & Inference

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

Network Archetecture
<img width="1410" height="758" alt="Screenshot 2025-10-01 092756" src="https://github.com/user-attachments/assets/00cf498b-b10e-4966-80d3-c94169b25e9a" />

