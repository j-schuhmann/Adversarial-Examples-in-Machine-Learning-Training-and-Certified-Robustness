
import tensorflow as tf

class attack:
    def __init__(self, model, epsilon, loss_func):

        self.model = model
        self.epsilon = epsilon
        self.loss_func = loss_func



    def perturb(self, x_nat, y):

        "perturbs the image with the respective attack"


class PGDAttack(attack):
    def __init__(self, model, epsilon, loss_func,iterations_attack,step_size_attack):

        super().__init__(model, epsilon, loss_func)
        self.step_size_attack = step_size_attack
        self.iterations_attack = iterations_attack

    
    def perturb(self, x_nat, y):

      
        x_adv = tf.identity(x_nat)
        for k in range(x_nat.shape[0]):
            x= tf.expand_dims(x_nat[k], axis=0)
            x_start = tf.identity(x)
            y_att= tf.expand_dims(y[k], axis=0)


            for i in range(self.iterations_attack):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    loss = self.loss_func(y_att, self.model(x))
                
                grad = tape.gradient(loss, x)
                x = x + self.step_size_attack * tf.sign(grad)
                x = tf.clip_by_value(x, x_start- self.epsilon, x_start+ self.epsilon)
                x = tf.clip_by_value(x, 0, 1)  # ensure valid pixel range
            
            x_adv = tf.tensor_scatter_nd_update(x_adv, [[k]], x)

        return x_adv
    

class PGDAttackl2(attack):
      

    def __init__(self, model, epsilon, loss_func,iterations_attack,step_size_attack):

        super().__init__(model, epsilon, loss_func)
        self.step_size_attack = step_size_attack
        self.iterations_attack = iterations_attack
        

    
    def perturb(self, x_nat, y):



        x_adv = tf.identity(x_nat)
        for k in range(x_nat.shape[0]):
            x= tf.expand_dims(x_nat[k], axis=0)
            x_start = tf.identity(x)
            y_att= tf.expand_dims(y[k], axis=0)

            for i in range(self.iterations_attack):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    loss = self.loss_func(y_att, self.model(x))
                
                grad = tape.gradient(loss, x)
                
                x = x + self.step_size_attack * grad / (tf.norm(grad,ord=2)+ 1e-12)

                diff = x - x_start
                diff_norm = tf.norm(diff, ord=2)
                scaling_factor = tf.where(diff_norm > self.epsilon, self.epsilon / (diff_norm + 1e-12), tf.ones_like(diff_norm))
        
        
                diff = diff * scaling_factor
            

                
                x = x_start+ diff



                x = tf.clip_by_value(x, 0, 1)  # ensure valid pixel range

            x_adv = tf.tensor_scatter_nd_update(x_adv, [[k]], x)

            return x_adv
                
      

class FGSMAttack(attack):
    def __init__(self, model, epsilon, loss_func):
            
        super().__init__(model, epsilon, loss_func)
         


    def perturb(self, x_nat, y):


        x_adv = tf.identity(x_nat)
        for k in range(x_nat.shape[0]):
            x= tf.expand_dims(x_nat[k], axis=0)

            y_att= tf.expand_dims(y[k], axis=0)


        
        
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = self.loss_func(y_att, self.model(x))
            
            grad = tape.gradient(loss, x)
            x = x + self.epsilon * tf.sign(grad)
            
            x = tf.clip_by_value(x, 0, 1) 
        
        x_adv = tf.tensor_scatter_nd_update(x_adv, [[k]], x)
            

        return x_adv
    

