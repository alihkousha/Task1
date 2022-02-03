import os
import tensorflow as tf
from typing import Dict, Text


class ClassProductModel(tf.keras.Model):

  def __init__(self, task : tf.keras.layers.Layer, user_model : tf.keras.Model, product_model : tf.keras.Model):
    super().__init__()
    self.product_model = product_model
    self.user_model = user_model
    self.task = task

  def train_step(self, features : Dict[Text, tf.Tensor]) -> tf.Tensor:


    with tf.GradientTape() as tape:


      user_embeddings = self.user_model(features["user_id"])
      positive_product_embeddings = self.product_model(features["product_id"])
      loss = self.task(user_embeddings, positive_product_embeddings)


      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:


    user_embeddings = self.user_model(features["user_id"])
    positive_product_embeddings = self.product_model(features["product_id"])
    loss = self.task(user_embeddings, positive_product_embeddings)


    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics