import os
import time
import logging
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse
from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# Configure logging
if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/azure_ai_image_generator.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Azure AI Image Generator startup')

class AzureAIImageGenerator:
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = '2023-06-01-preview'
        self.timeout = int(os.getenv('GENERATION_TIMEOUT', 300))
        self.poll_interval = int(os.getenv('POLL_INTERVAL', 5))
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_image(self, prompt):
        """Generate image from text prompt using Azure AI Foundry"""
        try:
            # Validate prompt
            if not prompt or len(prompt.strip()) < 5:
                raise ValueError("Prompt must be at least 5 characters long")

            # Prepare API request
            url = f"{self.endpoint}/openai/images/generations:submit?api-version={self.api_version}"
            payload = {
                "prompt": prompt,
                "size": os.getenv('IMAGE_SIZE', '1024x1024'),
                "n": 1,
                "quality": os.getenv('IMAGE_QUALITY', 'standard')
            }

            app.logger.info(f"Submitting image generation request for prompt: {prompt[:50]}...")
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()

            operation_location = response.headers.get('operation-location')
            if not operation_location:
                raise ValueError("No operation-location header in response")

            return self._poll_for_result(operation_location)

        except requests.exceptions.RequestException as e:
            app.logger.error(f"API request failed: {str(e)}")
            raise RuntimeError("Failed to connect to image generation service")
        except Exception as e:
            app.logger.error(f"Image generation failed: {str(e)}")
            raise

    def _poll_for_result(self, operation_url):
        """Poll for generation result until completion or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                response = requests.get(operation_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                status_data = response.json()

                if status_data['status'] == 'succeeded':
                    image_url = status_data['result']['data'][0]['url']
                    if self._validate_image_url(image_url):
                        app.logger.info("Image generation succeeded")
                        return image_url
                    raise ValueError("Invalid image URL received")

                elif status_data['status'] in ['failed', 'canceled']:
                    error_msg = status_data.get('error', {}).get('message', 'Unknown error')
                    raise RuntimeError(f"Generation failed: {error_msg}")

                time.sleep(self.poll_interval)

            except requests.exceptions.RequestException as e:
                app.logger.warning(f"Polling request failed: {str(e)}")
                time.sleep(self.poll_interval * 2)  # Backoff on failure
                continue

        raise TimeoutError("Image generation timed out")

    def _validate_image_url(self, url):
        """Validate that the URL is from our Azure endpoint"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and self.endpoint in url
        except:
            return False

# Initialize Azure AI client
azure_client = AzureAIImageGenerator(
    endpoint=os.getenv('AZURE_ENDPOINT'),
    api_key=os.getenv('AZURE_API_KEY')
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt', '').strip()
        try:
            if not prompt:
                flash('Please enter a prompt', 'error')
                return redirect(url_for('index'))

            image_url = azure_client.generate_image(prompt)
            session['image_url'] = image_url
            session['prompt'] = prompt
            return redirect(url_for('result'))

        except ValueError as e:
            flash(str(e), 'error')
        except RuntimeError as e:
            flash(str(e), 'error')
            app.logger.error(f"Generation error: {str(e)}")
        except Exception as e:
            flash('An unexpected error occurred', 'error')
            app.logger.exception("Unexpected error during generation")

    return render_template('index.html')

@app.route('/result')
def result():
    image_url = session.get('image_url')
    prompt = session.get('prompt')
    
    if not image_url:
        flash('No generated image found', 'error')
        return redirect(url_for('index'))
    
    return render_template('result.html', image_url=image_url, prompt=prompt)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error=error), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server error: {str(error)}")
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # This is only for development
    app.run(host='0.0.0.0', port=5000)
