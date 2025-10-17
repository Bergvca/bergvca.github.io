---
layout: post
title:  "Yet Another LLM Wrapper"
date:   2025-03-01
---

<img src="/media/img/podcast_generator.png" style="height: 100%;width: 100%;">

## Yet Another LLM Wrapper

So I [vibe-coded](https://x.com/karpathy/status/1886192184808149383) a LLM wrapper: [podcast generator](https://bergvca.github.io/site_to_pod/). 

The web-app takes in a URL (for example: a wikipedia page) and returns a podcast based on the website. 

What the point is?

Well, I'm not sure... 

It's a fun little project and maybe it can be of some use to someone, for example, for studying a subject while doing the dishes
or to make a website more accessible.

## How it's made

The project started out as a study project to learn more about Azure, Generative AI and React Native. 

It's not necessarily tied to Azure though: The backend runs in a Docker container, and the models used could just 
as well be huggingface or other models, however, for billing purposes it's very convenient to use a single platform. 

For the front-end I choose React Native. The main reason I choose React (as a non-ui developer) is that it seems to be the most popular 
framework for mobile development. The popularity of a language is increasingly more important in the age of AI: it means
more training data which results in more accurate models, which means total front-end n00bs such as myself can just ask 
a model to build a UI for them (and get a shitload of spaghetti code in return).  

### Frontend -  Expo

The frontend is built using [Expo](https://expo.dev/), which is a framework for React Native.
The nice thing about Expo is that it allows you to run the app on your phone without having to build a native app. In 
the end I exported it as a webpage and hosted it on github pages, but it can easily be compiled to a native app to run on 
Android or iOS. 

### Backend - FastAPI 

The backend is a [FastAPI](https://fastapi.tiangolo.com/) api. It tries to extract the main content of the website and 
feeds it to an LLM. The prompt of the LLM is as follows (where `content` is the main content of the website):
```python
[
    {
        "role": "system",
        "content": "You are two funny podcasts hosts. The first is very sarcastic and humoristic and is called Anton"
        "The second is more informative and an expert on the topic and is called Berta."
        "Before each sentence, add the name of the host who is speaking (Anton or Berta)."
        "Don't explain your reasoning at the beginnig of the output, start with the start of the podcast. "
        "Don't mention like or subscribe to this podcast."
        "Don't laugh.",
    }, {
        "role": "user",
        "content": f"create a podcast on the following article: {content}. Make sure the two hosts are present."
    }
]
```
This returns a text that can be split on the podcast host (Anton or Berta). These lines are then fed (in parallel) to an
Text-to-Speech (TTS) model, which returns a set of audio files. The audio files are then combined into a single mp3 file 
using ffmpeg and served to the user. 

## "_Vibe Coding_"

Alot has been written about LLM assisted coding, so I won't go into to much details here. The fact is that I was able to
build a (basic) UI with very little knowledge of React Native. The code could very well be described as "_spaghetti code_"
but I've been told that this is normal for React :). 

## Next steps

Please try out the [project](https://bergvca.github.io/site_to_pod/) if you are intrested. I don't plan on continuing 
development on this project, but I'm open to suggestions for improvement and additional functionality if there is enough interest. 