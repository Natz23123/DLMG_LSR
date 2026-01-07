import photo_to_vectors 
import training_vectors

def train():
    photo_to_vectors.photo_to_vectors()
    training_vectors.train_vectors()

train()