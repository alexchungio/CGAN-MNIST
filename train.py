#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/2 下午2:01
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import time
import tensorflow as tf

from libs.configs import cfgs
from data.dataset_pipeline import load_mnist, generate_dataset, sample_label
from libs.nets.model import Generator, Discriminator
from utils.image_utils import save_images

generator = Generator(cfgs.Y_DIM)
discriminator = Discriminator(y_dim=cfgs.Y_DIM, dropout_rate=cfgs.DROPOUT_RATE)
#
# generated_image = generator(sample_noise)
# # show_image_grid(generated_image, batch_size=cfgs.BATCH_SIZE)
# # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# show_image_grid(generated_image)
#
# decision = discriminator(generated_image)
# print(decision)


# ---------------------- loss function-----------------------------

# Define loss functions and optimizers for both models.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# discriminator loss
def discriminator_loss(real_output, fake_output):
    """

    :param real_output: (batch_size, 1)
    :param fake_output: (batch_size, 1)
    :return:
    """
    d_real_loss = cross_entropy(y_true=tf.ones_like(real_output, dtype=tf.float32),
                              y_pred=real_output)

    d_fake_loss = cross_entropy(y_true=tf.zeros_like(fake_output, dtype=tf.float32),
                              y_pred=fake_output)
    d_loss = d_real_loss + d_fake_loss

    return d_loss


# generator loss
def generator_loss(fake_output):
    """
    The generator's loss quantifies how well it was able to trick the discriminator
    :param fake_output: (batch_size, 1)
    :return:
    """
    g_fake_loss = cross_entropy(y_true=tf.ones_like(fake_output, dtype=tf.float32),
                                  y_pred=fake_output)

    g_loss = g_fake_loss

    return g_loss


# ----------------------------------optimizer--------------------------------
generator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, learning_rate=cfgs.GENERATOR_LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(beta_2=0.5, learning_rate=cfgs.DISCRIMINATOR_LEARNING_RATE)

# ----------------------------------trian log---------------------------------------
# checkpoint
ckpt = tf.train.Checkpoint(generator=generator,
                           discriminator=discriminator,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=cfgs.TRAINED_CKPT, max_to_keep=5)

# --------------------------train start with latest checkpoint----------------------------
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)

summary_writer = tf.summary.create_file_writer(cfgs.SUMMARY_PATH)
# -------------------------------train step---------------------------------------
# @tf.function
def train_step(real_image, label, noise):
    """

    :param real_image:
    :param fake_image:
    :return:
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(noise, label, training=True)

        real_output = discriminator(real_image, label, training=True)
        fake_output = discriminator(generated_image, label, training=True)

        gen_loss = generator_loss(fake_output=fake_output)
        disc_loss = discriminator_loss(real_output=real_output, fake_output=fake_output)

        gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))

    return gen_loss, disc_loss


# --------------------------------- train-------------------------------
def train(dataset, epochs):
    """

    :param dataset:
    :param epochs:
    :return:
    """
    global_step = 0
    for epoch in range(epochs):

        start_time = time.time()
        epoch_steps = 0
        gen_losses = 0.
        disc_losses = 0.
        for (batch, (image_batch, label_batch)) in enumerate(dataset):

            # Get the z
            batch_z = tf.random.uniform(minval=-1, maxval=1, shape=[cfgs.BATCH_SIZE, cfgs.Z_DIM])

            gen_loss, disc_loss = train_step(image_batch, label_batch, batch_z)

            gen_losses += gen_loss
            disc_losses += disc_loss

            epoch_steps += 1
            global_step += 1

            if (batch + 1) % cfgs.SHOW_TRAIN_INFO_INTE == 0:
                print('Epoch {} Batch {} Generator Loss {:.4f} Discriminator Loss {:.4f}'.format(
                    epoch + 1, batch, gen_loss / batch, disc_losses/ batch))

            if global_step % cfgs.SMRY_ITER == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('generator_loss', (gen_losses / epoch_steps), step=global_step)
                    tf.summary.scalar('discriminator_loss', (disc_losses / epoch_steps), step=global_step)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print('Epoch {} Generator Loss {:.4f} | Discriminator Loss {:.4f}'.format(epoch + 1,
                                                                                gen_losses / epoch_steps,
                                                                                disc_losses / epoch_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))


        sample_z = tf.random.uniform(minval=-1, maxval=1, shape=[10*10, cfgs.Z_DIM])
        sample_y = sample_label()
        generated_image = generator(z=sample_z, y=sample_y, training=False)

        save_images(generated_image, size=(10, 10),
                    image_path=os.path.join(cfgs.IMAGE_SAVE_PATH, 'epoch_{:04d}.png'.format(epoch)))


def main():
    # load mnist
    images, labels = load_mnist()
    dataset = generate_dataset(images, labels, batch_size=cfgs.BATCH_SIZE)
    train(dataset, cfgs.NUM_EPOCH)

    # generate_gif(image_path=cfgs.IMAGE_SAVE_PATH,
    #              anim_file=os.path.join(cfgs.IMAGE_SAVE_PATH, 'dcgan_mnist_dropout_05.gif'))


if __name__ == "__main__":
    main()

