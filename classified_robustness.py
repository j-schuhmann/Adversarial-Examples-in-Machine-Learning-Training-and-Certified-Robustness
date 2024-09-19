import tensorflow as tf
from scipy.optimize import brentq
import numpy as np
from attacks import PGDAttack
from models import ground_model_delta_experiments1, ground_model_delta_experiments2, ground_model_delta_experiments3, ground_model_delta_experiments4
from util import custom_sign, model_with_sign, positive_negative_parts





def compute_bounds(model, x, delta,activation):
    
    
    
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    delta = tf.convert_to_tensor(delta, dtype=tf.float32)
    
    #add batch dimension:
    if len(x.shape) <2:
        x= np.expand_dims(x, axis=0)

    # Initialize
    x_curr = x
    
    delta_up = delta * tf.ones_like(x)
    delta_low = delta * tf.ones_like(x)
    layer_counter=0
    # Propagate through each layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            if layer_counter<len(model.layers)-1:
            
                
                W = layer.kernel
                b = layer.bias
                
                
                
                W_positive, W_negative = positive_negative_parts(W)
                x_next = activation(tf.matmul(x_curr,W) + b)
                delta_up_new=activation(tf.matmul(x_curr,W)+tf.matmul(delta_up,W_positive)+tf.matmul(delta_low,W_negative)+b)-x_next
                delta_low_new=-activation(tf.matmul(x_curr,W)-tf.matmul(delta_low,W_positive)-tf.matmul(delta_up,W_negative)+b)+x_next

                x_curr=x_next
                delta_up=delta_up_new
                delta_low=delta_low_new

                layer_counter+=1
            
            else:
            
                W = layer.kernel
                b = layer.bias
                W_positive, W_negative = positive_negative_parts(W)

                x_final = tf.matmul(x_curr,W) + b
               
                delta_up_new = tf.matmul(delta_up,W_positive)+tf.matmul(delta_low,W_negative)
                delta_low_new=tf.matmul(delta_low,W_positive)+tf.matmul(delta_up,W_negative)
                
                delta_up=delta_up_new
                delta_low=delta_low_new


        else:
            
            layer_counter+=1
            
    x_delta_up=x_final + delta_up
    x_delta_low=x_final - delta_low     
    return x_final.numpy().flatten(),x_delta_up.numpy().flatten(), x_delta_low.numpy().flatten()         





def find_epsilon_max(model,x, max_interval, max_iterations,activation=tf.nn.sigmoid):
    gamma=1e-7 #to make sure we are not equal to 0 

    def delta_up_function(delta):
        _,delta_up,_=compute_bounds(model, x, delta,activation)
        return delta_up
    
    def delta_low_function(delta):
        _,_,delta_low=compute_bounds(model, x, delta,activation)
        return delta_low
    
    if delta_up_function(0)<0 and delta_low_function(0)<0:
        zero_delta_up = brentq(delta_up_function, 0, max_interval, maxiter=max_iterations)
        zero_delta_up=zero_delta_up-gamma       # so we are not exactly 0 

        if delta_up_function(zero_delta_up)*delta_low_function(zero_delta_up)>0:
            return zero_delta_up
        else:
            error_message = "Error: delta_up_function(zero_delta_up)*delta_low_function(zero_delta_up) is negative!."
            raise ValueError(error_message)

        
    if delta_up_function(0)>0 and delta_low_function(0)>0:
        zero_delta_low= brentq(delta_low_function, 0, max_interval)
        zero_delta_low=zero_delta_low-gamma

        if delta_up_function(zero_delta_low)*delta_low_function(zero_delta_low)>0:
            return zero_delta_low
        else:
            error_message = "Error: delta_up_function(zero_delta_low)*delta_low_function(zero_delta_low) is negative!."
            print(delta_up_function(zero_delta_low),delta_low_function(zero_delta_low))
            raise ValueError(error_message)


    else:
        error_message = "Error: delta_up_function(0) and delta_low_function(0) have different signs!."
        raise ValueError(error_message)
    


def compute_delta_test_set(model, x_test,max_interval, max_iterations):
    delta_array = []
    for k in x_test:
        delta = find_epsilon_max(model, k.flatten(),max_interval, max_iterations)
        delta_array.append(delta)

    mean=np.mean(delta_array)
    medium=np.median(delta_array)
    max=np.max(delta_array)
    min=np.min(delta_array)

    return delta_array,mean,medium,max,min



    
def compare2models_trained(model1_trained, model2_trained, x_test, y_test):
    # compare the epsilon_max for different models
    model_with_sign_1 = tf.keras.models.Sequential([
        model1_trained,                       
        tf.keras.layers.Lambda(custom_sign)          
    ])

    model_with_sign_2 = tf.keras.models.Sequential([
        model2_trained,                       
        tf.keras.layers.Lambda(custom_sign)          
    ])


    correct_indices_per_model = []
    trained_models_sign=[model_with_sign_1,model_with_sign_2]
    for model in trained_models_sign:
        
        prediction = model.predict(x_test)
        
        
        correct_indices = np.where(prediction.flatten() == y_test)[0]
        
        
        correct_indices_per_model.append(set(correct_indices))


    common_correct_indices = set.intersection(*correct_indices_per_model)

    # we only want the images which are correctly classified by all models, as our algo works only for correct onse! 
    correct_classified_x = x_test[list(common_correct_indices)]



    mean=[]
    medium=[]
    max=[]
    min=[]

    for model in [model1_trained,model2_trained]:
        
        delta_array=[]

        for k in correct_classified_x:
            delta=find_epsilon_max(model, k.flatten(),max_interval=1, max_iterations=100)
            delta_array.append(delta)
    
        mean.append(np.mean(delta_array))
        medium.append(np.median(delta_array))
        max.append(np.max(delta_array))
        min.append(np.min(delta_array))

    return  mean, medium, max, min

    
def test_bounds_attack(model,delta_array,gamma,step_size_attack,iterations_attack,x_test,y_test):
    # test the caluclated epsilon_max against the PGD attack, where the attack perturbation corresponds to the epsilon_max
    k=0
    for x,y,i in zip(x_test,y_test,delta_array):
        epsilon=i+gamma
        
        pgd=PGDAttack(model, epsilon=epsilon, random_start=True, loss_func=tf.keras.losses.MeanSquaredError(),iterations_attack=iterations_attack,step_size_attack=step_size_attack)
        adv_image = pgd.perturb(x,y)

        prediction=model_with_sign(model).predict(x_test)
        prediction_adv = model_with_sign(model).predict(adv_image)
        if prediction==y and prediction_adv!=y:
            print('attack successful')
            k+=1

    return k
            
        
    

# the followoing function is not up to date and should be updated

def test_iterations_influence(model,x_train, y_train, x_test, y_test, max_epochs, batch_size,verbose,evalute_every):
    """testing the influence of the number of iterations/loss/acc on the delta"""
    loss=[]
    acc=[]
    trained_models=[]
    trained_models_sign=[]

    correct_classified_x= x_test
    
    
    input_shape = (28, 28, 1)

    for i in range(1, max_epochs + 1):
        if i % evalute_every == 0:
            modelnew=ground_model_delta_experiments1(input_shape)
            loss_iter,acc_iter,trained_model,trained_model_sign=train_model(modelnew, x_train, y_train, x_test, y_test, i, batch_size,verbose)
            loss.append(loss_iter)
            acc.append(acc_iter)

            trained_models.append(trained_model)
            trained_models_sign.append(trained_model_sign)


    correct_indices_per_model = []


    for model in trained_models_sign:
       
        prediction = model.predict(x_test)
        
        
        correct_indices = np.where(prediction.flatten() == y_test)[0]
        

        correct_indices_per_model.append(set(correct_indices))


    common_correct_indices = set.intersection(*correct_indices_per_model)

    # we only want the images which are correctly classified by all models, as our algo works only for correct onse! 
    correct_classified_x = x_test[list(common_correct_indices)]
    


    mean=[]
    medium=[]
    max=[]
    min=[]

    for i in range(len(trained_models)):
        
        delta_array=[]

        for k in correct_classified_x:
            delta=find_max_delta(trained_models[i], k.flatten(),max_interval=1, max_iterations=50)
            delta_array.append(delta)
    
        mean.append(np.mean(delta_array))
        medium.append(np.median(delta_array))
        max.append(np.max(delta_array))
        min.append(np.min(delta_array))

    return  mean, medium, max, min, loss, acc

