import tensorflow as tf
from  attacks import PGDAttack,FGSMAttack,PGDAttackl2
from models import ground_model
import numpy as np 
import json
import pickle
from util import model_with_sign,print_accuracy





def train_step(model, images, labels, loss_object, optimizer, attack,l2=False):

        if attack == None:
            adv_images = images
        else:
            adv_images = attack.perturb(images, labels)
        
        with tf.GradientTape() as tape:
            predictions = model(adv_images)
            loss = loss_object(labels, predictions)
        



        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


class training_model():


    def __init__(self, model, optimizer, loss_object, epochs, batch_size, x_train, y_train, x_test, y_test):



        self.class_name = self.__class__.__name__
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.loss_array=None

        self.attack=None
        self.iterations_attack_train=None
        self.step_size_attack_train=None
        self.epsilon_attack_train=None



    def train(self):

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(50000).batch(self.batch_size)

        self.loss_array=[]
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            epoch_loss = 0
            
            for step, (images, labels) in enumerate(train_dataset):
                loss = train_step(self.model, images, labels,self.loss_object, self.optimizer,self.attack)
    
                epoch_loss += loss

            print(f'Loss: {epoch_loss.numpy()}')
            self.loss_array.append(epoch_loss.numpy())
            
    
    def attack_fgsm(self,epsilon_attack,robustness_experiment=None):
        
      

        attack=FGSMAttack(self.model, epsilon_attack, self.loss_object)
 
        adv_images = attack.perturb(self.x_test, self.y_test)

        adv_accuracy=print_accuracy(self.model, adv_images, self.y_test,adversarial=True,robustness_experiment=robustness_experiment)
        nat_accuracy=print_accuracy(self.model, self.x_test, self.y_test,robustness_experiment=robustness_experiment)
        return self.model, adv_accuracy,nat_accuracy
    
    def attack_pgd(self,epsilon_attack,iterations_attack,step_size_attack,robustness_experiment=None):

    
        attack=PGDAttack(self.model, epsilon_attack, self.loss_object,iterations_attack,step_size_attack)
        adv_images = attack.perturb(self.x_test, self.y_test)

        adv_accuracy=print_accuracy(self.model, adv_images, self.y_test,adversarial=True,robustness_experiment=robustness_experiment)
        nat_accuracy=print_accuracy(self.model, self.x_test, self.y_test,robustness_experiment=robustness_experiment)
        return self.model, adv_accuracy,nat_accuracy
    

    def attack_pgd_l2(self,epsilon_attack,iterations_attack,step_size_attack,robustness_experiment=None):
        
        
        attack=PGDAttackl2(self.model, epsilon_attack, self.loss_object,iterations_attack,step_size_attack)
        adv_images = attack.perturb(self.x_test, self.y_test)

        adv_accuracy=print_accuracy(self.model, adv_images, self.y_test,adversarial=True,robustness_experiment=robustness_experiment)
        nat_accuracy=print_accuracy(self.model, self.x_test, self.y_test,robustness_experiment=robustness_experiment)
        return self.model, adv_accuracy,nat_accuracy

    
       
    

    def evaluate_model_test_set(self,robustness_experiment=False):
        
        accuracy=print_accuracy(self.model, self.x_test, self.y_test,robustness_experiment=robustness_experiment)
        return accuracy

    def save_model_with_logs(self):
        # Save the entire class instance to a file and create log file
       
        
        path = f'saved_models/models/instance_{self.class_name}_epochs_{self.epochs}.pkl'
        with open(path, 'wb') as instance_file:
            pickle.dump(self, instance_file)



     
        model_config = self.model.to_json()



       
        parameters = {
            'class_name': self.class_name,
            'optimizer': str(self.optimizer),
            'loss_object': str(self.loss_object),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'x_train_shape': self.x_train.shape[0],
            'y_train_shape': self.y_train.numpy().shape[0],
            'x_test_shape': self.x_test.shape[0],
            'y_test_shape': self.y_test.numpy().shape[0],
            'epsilon__attack_train': self.epsilon_attack_train,
            'iterations_attack_train': self.iterations_attack_train,
            'step_size_attack_train': self.step_size_attack_train,
            'model_config': json.loads(model_config)  
        }

        
        log_file_path = f'saved_models/log_files/log_file_{self.class_name}.json'
        with open(log_file_path, 'w') as log_file:
            json.dump(parameters, log_file, indent=4)


    @staticmethod
    def load_model(path):
        # Load the class instance from a file
        with open(path, 'rb') as instance_file:
            return pickle.load(instance_file)

        
class pgd_adversarial_training(training_model):

    def __init__(self, model, optimizer, loss_object, epochs, batch_size, x_train, y_train, x_test, y_test,epsilon_attack_train,iterations_attack_train, step_size_attack_train):
        super().__init__(model, optimizer, loss_object, epochs, batch_size, x_train, y_train, x_test, y_test)
        
        self.iterations_attack_train = iterations_attack_train
        self.step_size_attack_train = step_size_attack_train
        self.epsilon_attack_train = epsilon_attack_train
        self.attack=PGDAttack(self.model, self.epsilon_attack_train,  self.loss_object,self.iterations_attack_train,self.step_size_attack_train)


class pgdl2_adversarial_training(training_model):

    def __init__(self, model, optimizer, loss_object, epochs, batch_size, x_train, y_train, x_test, y_test,epsilon_attack_train,iterations_attack_train, step_size_attack_train):
        super().__init__(model, optimizer, loss_object, epochs, batch_size, x_train, y_train, x_test, y_test)
        
        self.iterations_attack_train = iterations_attack_train
        self.step_size_attack_train = step_size_attack_train
        self.epsilon_attack_train = epsilon_attack_train
        self.attack=PGDAttackl2(self.model, self.epsilon_attack_train, self.loss_object,self.iterations_attack_train,self.step_size_attack_train)




        
class fgsm_adversarial_training(training_model):
    def __init__(self, model, optimizer, loss_object, epochs, batch_size, x_train, y_train, x_test, y_test,epsilon_attack_train):

        super().__init__(model, optimizer, loss_object, epochs, batch_size, x_train, y_train, x_test, y_test)
        


        self.epsilon_attack_train = epsilon_attack_train

        self.iterations_attack_train = None
        self.step_size_attack_train = None

        self.attack=FGSMAttack(self.model, self.epsilon_attack_train, self.loss_object)



        

    