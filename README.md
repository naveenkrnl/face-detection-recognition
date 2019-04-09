# face-detection-recognition

A Face Detection and Recognition Application using various modules such as opencv, numpy, etc.

Download the project or clone it to a directory
Create a virtualenv using 
<pre>virtualenv python3 -p venv</pre>

Now install all the dependencies using
<pre>pip install -r requirments.txt</pre>

Now to add a person to be recognized by the app
Construct a folder named "s<strong>n</strong>"(where n is a number) Example - s1,s2,s3,etc. in training-data and add images of the person you want to match with

Now open main.py
and add name of the person with images in the list <strong>"subject_names"</strong> at nth position

Bingo Now run the app using 
<pre>python main.py</pre>
