import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model
import numpy as np
import cv2
import os
import sklearn
import random
import pickle
from logger import get_loger
from models import generator, discriminator
from dataset import smooth

class Train:
    def __init__(self, 
                cartoon_data_directory, 
                original_data_directory,
                vgg_weights_directory,
                log_dir = None,
                checkpoint_dir = None,
                g_checkpoint_prefix = None,
                d_checkpoint_prefix = None,
                pretrain_checkpoint_prefix = None,
                epoch_save = 10,
                initialization_epoch = 5,
                gan_epoch = 10,
                batch_size_cartoon = 20, 
                batch_size_original = 10, 
                init_learning_rate = 2e-5,
                gen_learning_rate = 8e-5,
                disc_learning_rate = 3e-5,
                content_lambda = 0.5,
                g_adv_lambda = 1):

        self.init_learning_rate = init_learning_rate
        self.gen_learning_rate = gen_learning_rate
        self.disc_learning_rate = disc_learning_rate
        self.batch_size_cartoon = batch_size_cartoon
        self.batch_size_original = batch_size_original
        self.cartoon_data_directory = cartoon_data_directory
        self.original_data_directory = original_data_directory
        self.content_lambda = content_lambda
        self.g_adv_lambda = g_adv_lambda
        self.initialization_epoch = initialization_epoch
        self.gan_epoch = gan_epoch
        self.epoch_save = epoch_save
        self.len_cartoon = len(os.listdir(cartoon_data_directory))
        self.len_original = len(os.listdir(original_data_directory))

        self.checkpoint_dir = checkpoint_dir
        self.pretrain_checkpoint_prefix = pretrain_checkpoint_prefix
        self.g_checkpoint_prefix = g_checkpoint_prefix
        self.d_checkpoint_prefix = d_checkpoint_prefix

        self.pretrain_checkpoint_prefix = os.path.join(checkpoint_dir, "pretrain", self.pretrain_checkpoint_prefix)
        self.g_checkpoint_dir = os.path.join(checkpoint_dir, self.g_checkpoint_prefix)
        self.d_checkpoint_dir = os.path.join(checkpoint_dir, self.d_checkpoint_prefix)
        self.g_checkpoint_prefix = os.path.join(self.g_checkpoint_dir, self.g_checkpoint_prefix)
        self.d_checkpoint_prefix = os.path.join(self.d_checkpoint_dir, self.d_checkpoint_prefix)

        self.logger = get_loger('__train__',log_dir)
        
        self.disc_model = discriminator()
        self.gen_model = generator()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.mae = tf.keras.losses.MeanAbsoluteError()

        self.cartoon_arr = list(range(1,self.len_cartoon))
        self.original_arr = list(range(1,self.len_original))
        random.shuffle(self.cartoon_arr)
        random.shuffle(self.original_arr)

        base_model = tf.keras.applications.VGG19(weights = "imagenet", include_top = False, input_shape = (256, 256, 3))
        vgg_output = base_model.get_layer("block4_conv3").output
        output = Conv2D(512, (3, 3), activation = 'linear', padding = 'same', name = 'block4_conv4')(vgg_output)
        self.vgg = Model(inputs = base_model.input, outputs = output)
        self.vgg.load_weights(vgg_weights_directory, by_name = True)


    def create_batches(self):
        current_cartoon = 0
        current_original = 0

        while True:
            
            batch_cartoon, batch_original = [],[]

            for i in self.cartoon_arr[current_cartoon : current_cartoon + self.batch_size_cartoon]:
                cartoon_image = cv2.imread(self.cartoon_data_directory + str(i) + '.jpg')
                try:
                    cartoon_image.any() == None
                except:
                    continue
                cartoon_image = cv2.resize(cartoon_image, (256,256))/255.0
                batch_cartoon.append(cartoon_image)

            for j in self.original_arr[current_original : current_original + self.batch_size_original]:
                original_image = cv2.imread(self.original_data_directory + str(j) + '.jpg')
                try:
                    original_image.any() == None
                except:
                    continue
                original_image = cv2.resize(original_image, (256,256))/255.0
                batch_original.append(original_image)


            current_cartoon += self.batch_size_cartoon
            current_original += self.batch_size_original
            
            if (current_cartoon + self.batch_size_cartoon >= self.len_cartoon):
                current_cartoon = current_cartoon + self.batch_size_cartoon - self.len_cartoon

            if (current_original + self.batch_size_original >= self.len_original):
                current_original = current_original + self.batch_size_original - self.len_original

            batch_cartoon = np.array(batch_cartoon)
            batch_original = np.array(batch_original)

            
            ret = []
            ret.append(batch_cartoon)
            ret.append(batch_original)
            

            yield ret

    def initialization(self):
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.init_learning_rate)
        #if init_saved_weight_dir:
        #    self.gen_model.load_weights(init_saved_weight_dir)

        checkpoint = tf.train.Checkpoint(generator=self.gen_model, optimizer = optimizer)
        try:
            status = checkpoint.restore(tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, "pretrain")))
            status.assert_consumed()

            self.logger.info("Previous Checkpoint has been restored")

        except:
            self.logger.info("Checkpoint is not found. Training from scratch")

        batches = self.create_batches()

        for epoch in range(self.initialization_epoch):

            for step in range(int (np.ceil(self.len_original / self.batch_size_original))):

                cartoon_data, original_data = next(batches)

                ndim = len(original_data.shape)
                if not ndim == 4 :
                    original_data = np.expand_dims(original_data, axis = 0)
                with tf.GradientTape() as g_tape:

                    generated_images = self.gen_model(original_data)

                    latent_space_original = self.vgg(original_data)
                    latent_space_generated = self.vgg(generated_images)

                    content_loss = self.content_lambda * self.content_loss_func(latent_space_original, latent_space_generated)
                    
                gen_grads = g_tape.gradient(content_loss, self.gen_model.trainable_variables)

                optimizer.apply_gradients(zip(gen_grads, self.gen_model.trainable_variables))

                print("Epoch {}, Step: {}, Loss: {} ".format(epoch + 1, step + 1, content_loss))
                if (step+1) % 30 == 0:
                    cv2.imshow(original_data[0] * 255.0)
                    cv2.imshow(generated_images[0].numpy() * 255.0)
                

            if (epoch + 1) % 5 == 0:
                #self.gen_model.save_weights("init_{}_weights.h5".format(epoch+1))
                checkpoint.save(file_prefix = self.pretrain_checkpoint_prefix)


    def train_gan(self):
        
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate = self.gen_learning_rate)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate = self.disc_learning_rate)
        """
        if gen_saved_weight_dir :
            self.gen_model.load_weights(gen_saved_weight_dir)
            if disc_saved_weight_dir :
                self.disc_model.load_weights(disc_saved_weight_dir)"""
        
        g_checkpoint = tf.train.Checkpoint(g_model = self.gen_model, g_optimizer = gen_optimizer)
        try:
            status = g_checkpoint.restore(tf.train.latest_checkpoint(self.g_checkpoint_dir))
            status.assert_consumed()

            self.logger.info("Latest generator checkpoint has been restored")

        except:
            self.logger.info("NO generator checkpoint found. Trying to load initialization checkpoints...")

            try:
                status = g_checkpoint.restore(tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, "pretrain")))
                status.assert_consumed()

                self.logger.info("Successfully loaded pretrained checkpoints")

            except:
                self.logger.info("NO pretrained checkpoint found. Training from scratch...")

        d_checkpoint = tf.train.Checkpoint(d_model = self.disc_model, d_optimizer = disc_optimizer)
        try:
            status = d_checkpoint.restore(tf.train.latest_checkpoint(self.d_checkpoint_dir))
            status.assert_consumed()

            self.logger.info("Latest discriminator checkpoint has been restored")

        except:
            self.logger.info("NO discriminator checkpoint found. Training from scratch...")
                
        batches = self.create_batches()

        for epoch in range(self.gan_epoch):
            for step in range(int (np.ceil(self.len_original / self.batch_size_original))):
                
                cartoon_data, original_data = next(batches)
                smooth_data = smooth(cartoon_data[len(cartoon_data)//2:])
                cartoon_data = cartoon_data[:len(cartoon_data)//2]

                with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

                    real_output = self.disc_model(cartoon_data)
                    generated_images = self.gen_model(original_data)
                    fake_output = self.disc_model(generated_images)
                    smooth_output = self.disc_model(smooth_data)

                    total_discriminator_loss, real_loss, fake_loss, smooth_loss, generator_loss = self.adversial_loss_func(real_output, fake_output, smooth_output)

                    content_loss = self.content_loss_func(self.vgg(original_data), self.vgg(generated_images))
                    total_generator_loss = self.content_lambda * content_loss + self.g_adv_lambda * generator_loss
                  
                disc_grads = d_tape.gradient(total_discriminator_loss, self.disc_model.trainable_variables)
                gen_grads = g_tape.gradient(total_generator_loss, self.gen_model.trainable_variables)

                disc_optimizer.apply_gradients(zip(disc_grads, self.disc_model.trainable_variables))
                gen_optimizer.apply_gradients(zip(gen_grads, self.gen_model.trainable_variables))

                print("Epoch: {}, Step: {}, Generator Loss: {}, Discriminator Loss: {}".format(epoch + 1, step + 1, total_generator_loss, total_discriminator_loss))
                if (step+1) % 30 == 0:
                    cv2.imshow(original_data[0] * 255.0)
                    cv2.imshow(generated_images[0].numpy() * 255.0)


            if (epoch+1) % self.epoch_save == 0:

                self.gen_model.save_weights("gen_{}_weights.h5".format(epoch+1))
                #self.disc_model.save_weights("disc_{}_weights.h5".format(epoch+1))
                g_checkpoint.save(self.g_checkpoint_prefix)
                d_checkpoint.save(self.d_checkpoint_prefix)


    def adversial_loss_func(self, real_output, fake_output, smooth_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        smooth_loss = self.cross_entropy(tf.zeros_like(smooth_output), smooth_output)
        generator_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        total_loss = real_loss + fake_loss + smooth_loss

        return total_loss, real_loss, fake_loss, smooth_loss, generator_loss


    def content_loss_func(self, original_images, pred_images):

        loss = self.mae(original_images, pred_images)

        return loss 

def main(**kwargs):

    mode = kwargs["mode"]
    trainer = Train(**kwargs)

    if mode == 'full':
        trainer.initialization()
        trainer.train_gan()

    elif mode == 'initialize':
        trainer.initialization()

    elif mode == 'train gan':
        trainer.train_gan()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['full', 'initialize', 'train gan'], default='full')
    parser.add_argument('--cartoon_data_dir', type=str, required=True)
    parser.add_argument('--original_data_dir', type=str, required=True)
    parser.add_argument('--vgg weights_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--generator_checkpoint_prefix', type=str, required=True)
    parser.add_argument('--discriminator_checkpoint_prefix', type=str, required=True)
    parser.add_argument('--pretrain_checkpoint_prefix', type=str, required=True)
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--initialization_epoch', type=int, default=1)
    parser.add_argument('--train_gan_epoch', type=int, default=10)
    parser.add_argument('--cartoon_data_batch_size', type = int, default=20)
    parser.add_argument('--original_data_batch_size', type=int, default=10)
    parser.add_argument('--initialization_learn_rate', type=int, default=2e-5)
    parser.add_argument('--generator_learn_rate', type=int, default=8e-5)
    parser.add_argument('--discriminator_learn_rate', type=int, default=4e-5)
    parser.add_argument('--content_lambda', type=int, default=10)
    parser.add_argument('--generator_adversarial_lambda', type=int, default=1)

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)