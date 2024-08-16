# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import google.generativeai as genai


genai.configure(api_key='AIzaSyCpa_B_yPMpmntT3YOkrVQ85cH9l7O5pIw')
model = genai.GenerativeModel('gemini-1.5-flash')


@blueprint.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':

        #Here is the moment when the processing is done
        dish_name = request.form.get('dishName')
        ingredients = request.form.get('ingredients')
        cuisine = request.form.get('inlineFormCuisineSelectPref')

        prompt_dish = f"""In a single line give me a healthier food than: {dish_name} of the same cuisine: {cuisine} . 
        You can only give food recommendation and nothing else. """
        healthier_dish = model.generate_content(prompt_dish).text
        prompt_description = f"""Give me a description of less than 50 words of the following dish:{healthier_dish}.
         You can only give a description of the food and nothing else"""
        dish_description = model.generate_content(prompt_description).text
        return render_template('home/index.html', segment='index', results=[healthier_dish, dish_description])
    else:
        return render_template('home/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


#Dev Note Here:
"""
We try to hard to load the and integrate the model, but We were facing so many issues.
Environment related and with the model Itself, due to incompatibility of the versions of the libraries.
I spent around 6 hours trying to figured a solution, but at the end It was not possible. 

"""


"""
class BertLayer(layers.Layer):
    def __init__(self, bert_model_name='bert-base-uncased', **kwargs):
        dtype = kwargs.pop('dtype', None)
        if dtype:
            self._dtype_policy = tf.keras.mixed_precision.Policy(dtype)
            kwargs['dtype'] = self._dtype_policy.name
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(bert_model_name)
        # Placeholder call to initialize the model's output shape
        self.dummy_input_ids = tf.constant([[0] * 128])
        self.dummy_attention_mask = tf.constant([[1] * 128])

    def build(self, input_shape):
        # Initialize the BERT model by making a dummy call
        self.bert(self.dummy_input_ids, attention_mask=self.dummy_attention_mask)
        super(BertLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_ids, attention_mask = inputs
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]

    def get_config(self):
        config = super(BertLayer, self).get_config()
        config.update({
            "bert_model_name": self.bert.name_or_path,
            "dtype": self._dtype_policy.name if hasattr(self, '_dtype_policy') else None
        })
        return config

    @classmethod
    def from_config(cls, config):
        dtype = config.pop('dtype', None)
        if dtype:
            config['dtype'] = tf.keras.mixed_precision.Policy(dtype)
        return cls(**config)



def rebuild_model(model_path):
    # Define input layers
    input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

    # Load the BERT model
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

    # Add additional layers for classification
    dropout = layers.Dropout(0.3)(bert_output)
    dense = layers.Dense(64, activation='relu')(dropout)
    output = layers.Dense(1, activation='sigmoid')(dense)

    rebuilt_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    rebuilt_model.load_weights(model_path)

    return rebuilt_model
"""

#model_path = 'apps/model/FoodRecommender.keras'
#model = tf.keras.models.load_model(model_path, custom_objects={'BertLayer': BertLayer})
