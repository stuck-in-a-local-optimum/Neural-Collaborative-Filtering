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
    Implementation of this WWW'17 conference's research paper on Nueral Collaborative Filtering architecture, which proposed use of deep-learning to fix the incompetence of traditional Matrix Factorization for a recommender-system!
    

    
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




<!-- <img width="283" alt="lrp-basic-rule" src="https://user-images.githubusercontent.com/55681180/176441874-724b94bf-1c74-4724-9dde-c6bf1dd484cf.png"> -->



Illustration of the LRP procedure:

<img width="650" alt="lrp-basic-rule" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/lrp_illustration.png">


Let j and k represent neurons in two successive layers, say layer ‘p’ and ‘q’ in the above neural network. The rule  to back- propagate relevance scores R_k from layer ‘q’ to neurons ‘j’ in the previous layer ‘q’  is the following:

 <p align="center">
<img width="200" alt="lrp-basic-rule" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/lrp-basic-rule.png">
 <p/>


Note: Here the index of upper summation, i.e., “j”  represents neurons of previous layer  “p” where relevance had to reach from neuron ‘k” of layer “q” by back propagating.

And  zjk represents the extent to which neuron j in the layer ‘n’ has contributed to the R_k, the relevance of neuron k from the layer ‘q’. The conservation property was maintained by the denominator term. Once the input features are reached, the propagation procedure ends.


Following are the three variants of LRP propagation rules:
__1) Basic LRP (LRP-0):__  This rule redistributes in proportion to the contributions of each input to the neuron activation as they occur in Eq.	


<img width="650" alt="lrp-basic-rule" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/lrp_rule0.png">

__2) Epsilon Rule (LRP-ε):__  A first enhancement of the basic LRP-0 rule consists of adding a small positive term “ε” in the denominator:
When the contributions to the activation of neuron k are weak or inconsistent, the role of “ε” is to absorb some importance. Only the most important explanation components survive absorption as they grow larger. This usually results in explanations that are less noisy and have fewer input features.

<img width="650" alt="lrp-rule-epsilon" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/lrp-rule-epsilon.png">




__3) Gamma Rule (LRP-γ):__ Another enhancement which the author have introduced is obtained by favoring the effect of positive contributions over negative contributions. 
 The parameter “ γ” determines how much positive contributions are preferred. As grows, negative influences begin to fade. The prevalence of positive contributions limits the size of positive and negative relevance that can grow throughout the propagation phase. This contributes to more consistent explanations.
 
 <img width="650" alt="lrp-rule-gamma" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/lrp-rule-gamma.png">
 <br />
 

<!-- Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#top">back to top</a>)</p>

 -->
### Built With

* [PyTorch](https://pytorch.org/)
* [XLM-Roberta](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)
* [HuggingFace](https://huggingface.co/)
<p align="right">(<a href="#top">back to top</a>)</p>

## Implementation
We first built a hate-speech detection model trained on Indian languages using XLM-Roberta from Hugging Face and used it to explain its prediction using the LRP described above.

The structure of LRP rules enables us to implement them in a simple and efficient manner and the implementation have  following major 4 steps:

<img width="650" alt="psuedo_code" src="https://github.com/stuck-in-a-local-optimum/Layerwise-Relevance-Back-Propagation/blob/master/images/psuedo_code.png">
<br/>


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

