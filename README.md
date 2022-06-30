<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Neural Collaborative Filtering</h3>

  <p align="center">
    Implementation of this WWW'17 conference's research paper on Neural Collaborative Filtering architecture, which proposed use of deep-learning to fix the incompetence of traditional Matrix Factorization for a recommender-system!
    

    
<br/>
    
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Paper</a>
    </li>
    <li><a href="#Implementation">Implementation</a></li>
    <li><a href="#Results">Results</a></li>
    <li><a href="#Applications">Applications</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE Paper -->
## Traditional MF

Consider the utility matrix in the following image:

<img width="650" alt="mf" src="https://github.com/stuck-in-a-local-optimum/Nueral-Collaborative-Filtering/blob/main/images/mf.png">

The entry 1 in the above utility matrix is observed interaction because we know that this user has interacted with this item before and for the 0 entry we call it unobserved interaction since we are not sure if this user will interact with this item in the future or not. <br/>

Now, What we want to know is among all the unobserved interaction items which of them the user are most likely to interact with.
<br/>

The traditional way to solve the recommender system problem is to decompose this user and item matrix aka unitarity matrix into two sub matrices: the user matrix and the item matrix. 
And for prediction we simply multiply these two sub matrices to reconstruct the utility matrix and the larger the value on these unobserved entries the more likely that the corresponding user is going to interact with the corresponding item.
This utility matrix is factorized in such a way that the loss or the difference between the reconstructed matrix and the true utility matrix is minimized.

## Problem in traditional MF

Essentially what matrix factorization does is that it projects each of the user and item onto a Latin space of size K so if user items are represented by K dimension Latent vectors we can measure the similarity between each Latent vector by computing a dot product.


In fact for prediction we're computing the dot product of each of the user Latin vectors and the item Latin vectors.
 However the paper argued that inner products limit the expressiveness of latent vectors.

To understand the problem, let us consider following image:
<img width="650" alt="mf_example" src="https://github.com/stuck-in-a-local-optimum/Nueral-Collaborative-Filtering/blob/main/images/mf_example.png">


Let us first focus on the top three rows of this utility matrix, by computing the cosine similarity between these vectors, we know that user-2 and user-3 are most similar  and user 1 and 3 are least similar. We now project these users onto the latent-space of dimension-2, since user-2 and user-3 are the most similar they are close to each other while user-1 and user-3 are least similar so they are far from each other.

 Now we consider user 4 while computing the similarity between user-4  and the others. We know that user 1 is the most similar with user-4 while user-2 is least similar.
 
 
 However here's where the problem comes in, no matter how we place user-4 around user-1 user 3 ends up being the farthest from user-4 while in reality user-2 is the most different from user-4 not user-3.W
 

The example shows the incompetence of inner product in bottling a complex interaction between user latent vectors and item latent vectors, so the paper proposed a new neural architecture calling it neural collaborative filtering.




## Implementation
__GENERALIZED MATRIX FACTORIZATION__

Following the GMF architecture proposed by authors. 

<img width="650" alt="gmf" src="https://github.com/stuck-in-a-local-optimum/Nueral-Collaborative-Filtering/blob/main/images/gmf.png">



We can see that  both the user and item are one hot encoded and then they are projected onto the latent space with an embedded layer.
 The neural CF layers basically can be any kind of neural connections, multiple layer perceptron for instance can be placed here.
 The paper claim that with the complicated connection in these layers and the non-linearity,  this model is capable of learning the user and item interactions in latent space properly.
 

<img width="650" alt="gmf2" src="https://github.com/stuck-in-a-local-optimum/Nueral-Collaborative-Filtering/blob/main/images/gmf2.png">
The authors also showed how matrix factorization is a special case of GMF,
if we replace the new CF layers here with a multiplication layer which performs element-wise product on its two inputs and if we also set the weights from the multiplication layer to the output layer to be a unity matrix and if we set the activation function of the output layer to be a linear function. Then this GMF becomes traditional MF.


__Neural Collaborative Filtering__

<img width="650" alt="ncf" src="https://github.com/stuck-in-a-local-optimum/Nueral-Collaborative-Filtering/blob/main/images/ncf.png">

So above is the final model proposed looks like this, it contains two sub modules, in order to introduce more non-linearity they includes a multi-layer perceptron model here in addition to original generalized material factorization layer and they fused these models with a concatenation layer followed by a sigmoid activation function.

We have implemented it in pytorch and have using Movie-lense dataset and book-crossing dataset for training and testing.


## Results

We used the following color spectrum to visualize our LRP explanations; the more the color is from the right side of the spectrum, the higher the contribution of that word to sentence label prediction (hate or not-hate).
<img width="650" alt="spectrum" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/spectrum.png">


__example1__: 
<br/>
<img width="650" alt="results_example1" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/results_example1.png">
<br/>

__example2__: 
<br/>
<img width="650" alt="results_example1" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/results_example2.png">


<!-- USAGE EXAMPLES -->
## Applications

LRP can be used for a variety of purposes, some of which are as follows:

Let’s say our network predicts a cancer diagnosis based on a mammogram (a breast tissue image), the explanation provided by LRP would be a map showing which pixels in the original image contribute to the diagnosis and to what amount. Because this approach does not interfere with network training, it can be used on already trained classifiers.

XML methods are especially useful in safety-critical domains where practitioners must know exactly what the network is paying attention to.  Other use-cases include network (mis)behavior diagnostics, scientific discovery, and network architectural improvement.



<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contacts
[LinkedIn](https://www.linkedin.com/in/ajeet-yadav-a507971a9/)
[Twitter](https://twitter.com/weightsNbiases)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [InDepth: Layer-Wise Relevance Propagation](https://towardsdatascience.com/indepth-layer-wise-relevance-propagation-340f95deb1ea)
* [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

