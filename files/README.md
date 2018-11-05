HTML5 Audio Query Template
==========================

A skeleton website built with Flask and HTML5 Audio, that can be used for building any query-by-humming-like service.

<img src="http://i.imgur.com/S2k8VH5.png" width="60%" alt="Screen Capture">

## Running

    pip install -r requirements.txt
    python runserver.py

## Hacking

### Backend

The web UI will upload a WAV file to the server; you can then process the file using your Python code [here](https://github.com/marl/html5-audio-query-template/blob/master/voice/views.py#L18-L28), and return the result as JSON.

### Frontend

The JSON object will be retrieved back to the browser, and you can edit [here](https://github.com/marl/html5-audio-query-template/blob/master/voice/templates/index.html#L10-L13) to change what to do with the result. By default, it shows the JSON in a textarea.

## Gotchas

Due to privacy issues, some browsers don't allow access to the microphone when using `http`. It'll be still okay to run this website locally, but you will need to obtain an SSL certificate if you'd like to publicly serve the website.
