## Anonymization Prompt Learning

### Abstract

Text-to-image diffusion models, such as Stable Diffusion, generate highly realistic images from text descriptions. However, the generation of certain content at such high quality raises concerns. A prominent issue is the accurate depiction of identifiable facial images, which could lead to malicious deepfake generation and privacy violations. In this paper, we propose Anonymization Prompt Learning (APL) to address this problem. Specifically, we train a learnable prompt prefix for text-to-image diffusion models, which forces the model to generate anonymized facial identities, even when prompted to produce images of specific individuals. Extensive quantitative and qualitative experiments demonstrate the successful anonymization performance of APL, which anonymizes any specific individuals without compromising the quality of non-identity-specific image generation. Furthermore, we reveal the plug-and-play property of the learned prompt prefix, enabling its effective application across different pretrained text-to-image models for transferrable privacy and security protection against the risks of deepfakes.

### Notes
This is an early release of the code for our paper Anonymization Prompt Learning for Facial Privacy-Preserving Text-to-Image Generation.

The code is not final, but should give a good starting point for anyone interested in the details of our methodology. 

README will be updated soon. 

