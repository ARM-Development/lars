# LLM-Assisted Radar Scene classification (LARS)

LARS uses a hybrid approach between multimodal LLMs (mLLMs) and deep learning computer vision models for radar scene classification.

LARS is based around Nepho, a parallel-mLLM chatbot interface designed by @thelechen to quickly prompt images to mLLMs using parallelism. LARS will consist of two portions:

* A module that uses mLLMs to automatically generate labelled radar images for supervised learning
* Modules for various fine tuned ImageNet/ViT models for classifying radar scenes.

Authors: Bobby Jackson, Le Chen, Scott Collis
