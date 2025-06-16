# Lab 3: Insights from Audio Transcription

---

## 📘 Introduction

In this lab, you'll dive into the world of audio transcription by taking a short meeting recording and turning it into text. You'll learn how to pull out the most important bits like the summary, key points, and action items, figure out the overall mood, and then wrap it all up by saving your polished meeting minutes as a text file. 


---

## 🎯 Learning Objectives

- **Understand the Capabilities of OpenAI's Speech Transcription Models**: Gain a foundational understanding of what OpenAI's speech transcription model Whisper is capable of, including strengths and limitations.
- **Develop Skills in Using OpenAI's API for Speech Transcription**: Learn to effectively utilize OpenAI's API to generate transcriptions, including setting up the necessary development environment, understanding API documentation, and executing API calls.
- **Creating a txt File**: Learn how to write the code for creating a txt file and store desired information.


---

## ⏱️ Estimated Time
3 hours

---

## 💵 Estimated Cost
$1 (View your OpenAI usage [here]((https://platform.openai.com/usage))

---

## 🧰 Materials

- OpenAI account & API key
- [Audio Transcription Colab](https://colab.research.google.com/drive/1AKP0Tw-TNUEbyOw_ykm9Xp4xW1g6kfVV?usp=sharing)
- [Meeting_Minutes.mp3](https://drive.google.com/file/d/1VuNnF4_EPLr1uVhdgK69206oDya88DvY/view?usp=sharing)

Picture

---

## ⚙️ 1.1 Setting Up Your Development Environment

1. Make a copy of the [Audio Transcription Colab](https://colab.research.google.com/drive/1AKP0Tw-TNUEbyOw_ykm9Xp4xW1g6kfVV?usp=sharing)
2. To set up your development environment, you will need to have Python installed. You will also need to install the OpenAI Python library, which we will use to generate text in this lab. You can run the following shell scripts, which are also pre-written in the first cell of your Colab notebook, to do so.

```Python
# Uninstall Colab's preinstalled modules which have dependency conflicts with
# the OpenAI Python library.
!pip uninstall tensorflow-probability -y
!pip uninstall llmx -y
# Install the OpenAI Python library.
!pip install --upgrade openai
```

3. We also want to create a folder to save the txt file for the text that we generate. Navigate to the “Files” section of the left sidebar by clicking on the file icon. In Files, you will see a folder named sample_data. Navigate to its parent folder by clicking the file icon which has an up arrow inside. Find the content folder, and within it create a new folder named **minutes**. 

(For more detailed information and screenshots of this process, refer to Steps 3-5 of the “Setting Up Your Development Environment” section of the Generating Images with OpenAI’s Models lab. Be sure to name your folder **minutes**.)

---


## 📥 1.2 Importing and Initializing OpenAI

1. Now that you have the OpenAI Python library installed, we can begin to code. At the top of your cell, import requests and OpenAI.

```Python
from openai import OpenAI
```


2. Next, initialize an OpenAI object. This will allow you to use the OpenAI API to communicate with OpenAI’s models. Make sure to replace your-api-key-here with your API key.


```Python
client = OpenAI(api_key='your-api-key-here')
```

---


## 🎙️ 2.1 Function to Transcribe the Audio

1. Create a function titled **transcribe_audio** with 1 parameter **audio_path_file**.
- This function will be used to transcribe the audio file using Whisper

```Python
def transcribe_audio(audio_path_file):
```

2. **audio.transcriptions.create** is a function from the OpenAI Python Library which generates a text transcription of an audio file it is given.

**open(audio_file_path, "rb")** opens the audio file using the **rb** mode. **r** for read and **b** for ‘binary’ indicates that the file has permission to be read in binary.


So far, OpenAI only has one model for speech recognition, Whisper, so we set the **model** to **whisper-1** as default. Since **audio.transcriptions.create** returns a Transcription object, which we’ve named **transcription**, we extract and return the transcribed text from the object with **transcription.text**.

```Python
# Given the path to an audio file, transcribes the audio using Whisper.
def transcribe_audio(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    return transcription.text
```

---

## 🧾 2.2 Function to Obtain a Summary

1. Create a function titled **abstract_summary_extraction** with 1 parameter **transcription**.
- This function will be used to prompt OpenAI to generate a summary of the transcription

```Python
# Takes the transcription of the meeting and returns a summary of it via text completions
def abstract_summary_extraction(transcription):
```

2. The **abstract_summary_extraction** function uses the **chat.completions.create** function from the **Generating Text with OpenAI’s Models lab** to have GPT generate a summary of the transcription. 

```Python
def abstract_summary_extraction(transcription):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content
```

---


## 📌 2.3 Function to Obtain Key Points


1. Create a function titled **key_points_extraction** with 1 parameter **transcription**.
This function will be used to extract key points from the transcribed audio

```Python
# Takes the transcription of the meeting and returns the key points in it via text completions 
def key_points_extraction(transcription):
```

2. The **key_points_extraction** function uses the **chat.completions.create** function from the **Generating Text with OpenAI’s Models** lab to give GPT the role of extracting the key points from the transcription.

```Python
def key_points_extraction(transcription):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content
```

---


## 🚀 2.4 Function to Obtain Action Items

1. Create a function titled **action_item_extraction** with 1 parameters **transcription**.
- This function will be used to define the action items from the transcription.

```Python
# Takes the transcription of the meeting and returns the action items from it via text completions 
def action_item_extraction(transcription):
```

2. The **action_item_extraction** function uses the **chat.completions.create** from the **Generating Text with OpenAI’s Models** lab to have GPT determine the action items within the transcript.

```Python
def action_item_extraction(transcription):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content
```

---





