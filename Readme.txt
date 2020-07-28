Note for Chris and JH:
Please forgive me that now the code still have a lot of redundant parts and parameters in each function, and I just
can't make sure the code can be run immediately. But one thing I want to make sure is to make it easier for you to
take a loot at the model. In model.py, there is 4 classes, MultiTiemStepLSTM, Siamese_Network_v2, VAE3d. For LSTM it's
the RNN classification model, for VAE3d it's actually can be either variational autoencoder, or just autoencoder. The
encoder and decoder part lies in base/models/nns/v13.py, you may take a look if you want. Please use loss_nkl and 
negatice_elbo_bound_ae in VAE3d to compute the reconstruction loss if you want it as a AE. 
the loss and negative_elbo_bound_cos is actually take it as VAE, feel free to ask me if you have any question.
For Siamese_Network_v2, this is what we proposed in LSSL, adding a consine loss. Siamese_Network_v2 is composed with
a AE(VAE) and a Dist_relu_loss class on top of the AE. Dist_relu_loss lies in base/models/nns/v13.py too, this class
contains a weight vector, which is actually the direction vector we want to find in LSSL. The Siamese_Network_v2 
will take a minibatch of pair images as their input, and feed it into VAE3d, after that, feed the output latent 
vector into Dist_relu_loss to compute the final loss.
You may also wonder which pairs to use, qingyu has chosen a list of them, which lies in img1.txt and img2.txt.
While each sample in img1.txt is the former in each pair and sample in img2.txt is the latter. You may see some 
repeated sample, this is because for each subjects, we may able to use repeated pairs to increase the acuracy.
 For instance, suppose subject s1 has 5 time points (t1,t2,t3,t4,t5), we may choose (t1,t2), (t1,t3),...,(t4,t5) into
consideration. In fact, qingyu chose those pairs with time gap more than 1 years, i.e, t2-t1>1 yr.
In test.txt, you may find the samples that we draw the the picture. Unfortunately, I couldn't give you the plot code
because the final result we used is drwan by qingyu. But you may able to reproduce the model and use these samples
see similar results.
For pre-training the model, it's actually not that hard. But you may find that I actually didn't contain a dropout
layer in the encoder and decoder where in qingyu's code there is. Both of them gives similar results, you may test
it if you want, qingyu's code is in /code_latest.  
If you feel it's hard to read my code, especially code w.r.t data processing part, I apologize for that. Because
I'm not a very good coding engineer. Here I want to quickly explain the how I generate data for classification.
For both cross sectional and longitudinal setting, we actually divided data into 5 folds, based on subject-level.
That means we can't split different time points for a single subject into different folds. For cross-sectional setting
data augmentation is based on iamge-level, while in longtudinal setting, data augmentation should based on subject-level.
Also, in cross-section setting we balance the total number of 2 different labels, while in longitudinal setting we 
balance the total number of subjects with 2 different labels. Suppose in fold 1, there is 100 AD subjects(400 images) 
and 120 NC subjects(460 images), then for cross-sectional setting, we use data augmentation to get 600 AD images and
6000 NC images. In longitudinal setting, we instead get 200 AD subjects and 200 NC subjects, this might lead to an 
imbalance on image-level, yet this wil not affect too much.
If you have any questions, please feel free to ask me ASAP, because your problem I might have faced before, maybe 
asking me is the best way for trouble-shooting.
I hope you can find some more interesting results, and I'm always ready to help!
BTW, you may find most functions in utils.py not that understandable, I apologize for that, feel free to contact me
and I will try to explain my intuition to you.

Best,
   Zucks