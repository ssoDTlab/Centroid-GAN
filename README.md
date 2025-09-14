## Feature Space-Based Center Constraint Loss for Improving Quality and Diversity in GANs(Centroid-GAN) <br> <sub>Official PyTorch implementation of the KOREAAI 2025 paper</sub>
<p align="center">
  <img width="380" height="400" alt="Image1" src="https://github.com/user-attachments/assets/5382e1c0-bf6d-4f85-8a03-a146083dd52a" />
  <img width="290" height="400" alt="Image2" src="https://github.com/user-attachments/assets/71fc1596-8deb-49e7-9ab9-5e8ee7cb2448" />
</p>

### <b>Abstract</b><br>
This paper proposes a novel constraint-based loss function to simultaneously improve the image quality and diversity of Generative Adversarial Networks (GANs). The proposed method first embeds the training dataset into a feature space using a pretrained neural network, and computes the centroid of the distribution. The average distance between each feature vector and the centroid is then defined as the reference distance. During training, generated images are also embedded into the same feature space, and if the feature vectors of generated images deviate from the centroid beyond the reference distance, a penalty is added to the loss function. This constraint guides the generator to produce images within the distributional range of real data, thereby effectively maintaining a balance between quality and diversity. Experiments conducted on the CelebA dataset demonstrate that the proposed approach outperforms conventional methods in terms of both FID and LPIPS metrics.

<br>
Paper Link : 
