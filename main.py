import os
import json
import time
from datetime import datetime

from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Configuration
OPENROUTER_API_KEY = "abcd"
LLM_MODEL_NAME = "google/gemini-2.0-pro-exp-02-05:free"  
ENABLE_CACHING = True
CACHE_DIR = "paper_cache"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Ensure cache directory exists
if ENABLE_CACHING:
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_paper(topic):
    """Retrieve paper from cache if available"""
    if not ENABLE_CACHING:
        return None
    topic_hash = str(hash(topic))
    cache_file = os.path.join(CACHE_DIR, f"{topic_hash}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                return cache_data.get('paper')
        except Exception as e:
            print(f"Cache read error: {str(e)}")
            return None
    return None

def save_to_cache(topic, paper):
    """Save paper to cache"""
    if not ENABLE_CACHING:
        return
    topic_hash = str(hash(topic))
    cache_file = os.path.join(CACHE_DIR, f"{topic_hash}.json")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'topic': topic, 'paper': paper, 'timestamp': datetime.now().isoformat()}, f, ensure_ascii=False)
    except Exception as e:
        print(f"Cache write error: {str(e)}")

def generate_research_paper(topic):
    """Generate research paper using direct API calls"""
    outline = generate_research_outline(topic)
    if not outline:
        raise Exception("Failed to generate research outline")
    latex_template = generate_latex_template(outline)
    if not latex_template:
        raise Exception("Failed to generate LaTeX template")
    paper_content = generate_paper_content(outline, latex_template)
    if not paper_content:
        raise Exception("Failed to generate paper content")
    paper_with_citations = enhance_citations(paper_content)
    if not paper_with_citations:
        return paper_content
    paper_with_diagrams = enhance_diagrams(paper_with_citations)
    if not paper_with_diagrams:
        return paper_with_citations
    final_paper = final_polish(paper_with_diagrams)
    if not final_paper:
        return paper_with_diagrams
    return final_paper

def call_openrouter_api(messages, temperature=0.7, max_tokens=4000):
    """Make a direct call to the OpenRouter API with improved error handling"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Research Paper Generator"
    }
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Sending payload: {json.dumps(payload, indent=2)}")
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            print(f"API call error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            print(f"Response content: {e.response.text}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise Exception(f"OpenRouter API error after {MAX_RETRIES} attempts: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response.text}")
            raise Exception(f"Error parsing JSON: {e}")
        except Exception as e:
            print(f"General API error: {e}")
            raise Exception(f"General API error: {e}")

def generate_research_outline(topic):
    """Generate research outline with direct API call"""
    prompt = f"You are a world-class academic researcher. Develop a comprehensive plan for a research paper on the topic: '{topic}'. Include title, abstract points, keywords, sections, diagrams, tables, mathematical areas, and key sources. Format as JSON."
    messages = [{"role": "system", "content": "You are a helpful research assistant."}, {"role": "user", "content": prompt}]
    try:
        response = call_openrouter_api(messages)
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except:
                pass
        try:
            return json.loads(response)
        except:
            return {"raw_outline": response}
    except Exception as e:
        print(f"Error generating research outline: {str(e)}")
        return None

def generate_latex_template(outline):
    """Generate LaTeX template with direct API call"""
    outline_str = json.dumps(outline, indent=2)
    prompt = f"You are an expert in LaTeX. Based on this research outline: {outline_str}, create a LaTeX template with document class, packages, styling, and bibliography style. Format as JSON."
    messages = [{"role": "system", "content": "You are a helpful LaTeX expert."}, {"role": "user", "content": prompt}]
    try:
        response = call_openrouter_api(messages)
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except:
                pass
        try:
            return json.loads(response)
        except:
            return {"raw_template": response}
    except Exception as e:
        print(f"Error generating LaTeX template: {str(e)}")
        return None

def generate_paper_content(outline, latex_template):
    """Generate paper content with direct API call"""
    outline_str = json.dumps(outline, indent=2)
    template_str = json.dumps(latex_template, indent=2)
    prompt = f"You are a world-class academic researcher. Write a complete research paper based on this outline: {outline_str} and LaTeX template: {template_str}. Return ONLY the complete LaTeX code."
    messages = [{"role": "system", "content": "You are a helpful research paper writer."}, {"role": "user", "content": prompt}]
    try:
        response = call_openrouter_api(messages, temperature=0.7, max_tokens=8000)
        import re
        latex_match = re.search(r'```(?:latex)?\s*([\s\S]*?)\s*```', response)
        if latex_match:
            return latex_match.group(1).strip()
        return response
    except Exception as e:
        print(f"Error generating paper content: {str(e)}")
        return None

def enhance_citations(paper_content):
    """Enhance citations with direct API call"""
    prompt = f"You are an expert in academic citation. Review this LaTeX paper: {paper_content} and enhance its citations and references. Return the complete LaTeX document."
    messages = [{"role": "system", "content": "You are a helpful citation expert."}, {"role": "user", "content": prompt}]
    try:
        response = call_openrouter_api(messages, temperature=0.7, max_tokens=8000)
        import re
        latex_match = re.search(r'```(?:latex)?\s*([\s\S]*?)\s*```', response)
        if latex_match:
            return latex_match.group(1).strip()
        return response
    except Exception as e:
        print(f"Error enhancing citations: {str(e)}")
        return None

def enhance_diagrams(paper_content):
    """Enhance diagrams with direct API call"""
    prompt = f"You are an expert in scientific visualization. Review this LaTeX paper: {paper_content} and enhance its diagrams and visualizations. Return the complete LaTeX document."
    messages = [{"role": "system", "content": "You are a helpful visualization expert."}, {"role": "user", "content": prompt}]
    try:
        response = call_openrouter_api(messages, temperature=0.7, max_tokens=8000)
        import re
        latex_match = re.search(r'```(?:latex)?\s*([\s\S]*?)\s*```', response)
        if latex_match:
            return latex_match.group(1).strip()
        return response
    except Exception as e:
        print(f"Error enhancing diagrams: {str(e)}")
        return None

def final_polish(paper_content):
    """Final polish with direct API call"""
    prompt = f"You are a meticulous academic editor and LaTeX expert. Review this LaTeX paper: {paper_content} and perform final polishing. Return the complete, finalized LaTeX document."
    messages = [{"role": "system", "content": "You are a helpful LaTeX editor."}, {"role": "user", "content": prompt}]
    try:
        response = call_openrouter_api(messages, temperature=0.7, max_tokens=8000)
        import re
        latex_match = re.search(r'```(?:latex)?\s*([\s\S]*?)\s*```', response)
        if latex_match:
            return latex_match.group(1).strip()
        return response
    except Exception as e:
        print(f"Error in final polish: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_paper():
    data = request.json
    topic = data.get('topic', '').strip()
    if not topic:
        return jsonify({'error': 'Please provide a research topic or title'}), 400
    cached_paper = get_cached_paper(topic)
    if cached_paper:
        return jsonify({'paper': cached_paper, 'cached': True})
    try:
        final_paper = generate_research_paper(topic)
        save_to_cache(topic, final_paper)
        return jsonify({'paper': final_paper, 'cached': False})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

def create_template_files():
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced AI Research Paper Generator</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; background-color: #f8f9fa; }
        h1 { text-align: center; margin-bottom: 30px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .container { background-color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 30px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #2c3e50; }
        .input-group { display: flex; margin-bottom: 20px; }
        input { flex-grow: 1; padding: 12px; border: 1px solid #ddd; border-radius: 4px 0 0 4px; font-size: 16px; transition: border-color 0.3s; }
        input:focus { outline: none; border-color: #3498db; box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2); }
        button { padding: 12px 24px; background-color: #3498db; color: white; border: none; border-radius: 0 4px 4px 0; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }
        button:hover { background-color: #2980b9; }
        button:disabled { background-color: #95a5a6; cursor: not-allowed; }
        .loading { text-align: center; padding: 40px 0; display: none; }
        .loading-spinner { border: 5px solid #f3f3f3; border-top: 5px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 2s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .stages { max-width: 600px; margin: 20px auto; padding: 0; list-style: none; }
        .stage { padding: 10px; margin-bottom: 8px; border-radius: 4px; background-color: #f1f1f1; color: #7f8c8d; position: relative; padding-left: 40px; transition: all 0.3s; }
        .stage::before { content: "○"; position: absolute; left: 15px; font-size: 18px; }
        .stage.active { background-color: #e1f0fa; color: #3498db; font-weight: 500; }
        .stage.active::before { content: "●"; color: #3498db; }
        .stage.completed { background-color: #e8f8f5; color: #27ae60; }
        .stage.completed::before { content: "✓"; color: #27ae60; }
        .output { margin-top: 30px; display: none; }
        .output-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .copy-btn { padding: 8px 16px; font-size: 14px; background-color: #2ecc71; color: white; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }
        .copy-btn:hover { background-color: #27ae60; }
        pre { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; padding: 15px; white-space: pre-wrap; overflow-x: auto; font-family: 'Courier New', monospace; font-size: 14px; max-height: 500px; overflow-y: auto; }
        .next-steps { background-color: #e8f0fe; border: 1px solid #d0e1f9; border-radius: 4px; padding: 20px; margin-top: 30px; }
        .next-steps h3 { color: #2c3e50; margin top: 0; border-bottom: 1px solid #d0e1f9; padding-bottom: 10px; }
        .next-steps ol { padding-left: 20px; }
        .next-steps li { margin-bottom: 8px; }
        .error { color: #e74c3c; background-color: #fadbd8; padding: 10px 15px; border-radius: 4px; margin-top: 10px; display: none; border-left: 4px solid #e74c3c; }
        footer { text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced AI Research Paper Generator</h1>
        <div>
            <label for="topic">Enter Research Topic or Title:</label>
            <div class="input-group">
                <input type="text" id="topic" placeholder="e.g., The Impact of Artificial Intelligence on Climate Change Mitigation">
                <button id="generate-btn">Generate Paper</button>
            </div>
            <div class="error" id="error-message"></div>
        </div>
        <div class="loading" id="loading">
            <div class="loading-spinner"></div>
            <p>Generating your high-quality research paper...</p>
            <p style="font-size: 14px; color: #666;">This process takes 3-5 minutes as our AI crafts a comprehensive academic paper.</p>
            <ul class="stages">
                <li class="stage" id="stage-1">Planning research structure and approach</li>
                <li class="stage" id="stage-2">Creating LaTeX template with appropriate packages</li>
                <li class="stage" id="stage-3">Generating detailed academic content</li>
                <li class="stage" id="stage-4">Enhancing citations and references</li>
                <li class="stage" id="stage-5">Improving diagrams and visualizations</li>
                <li class="stage" id="stage-6">Final polishing and quality assurance</li>
            </ul>
        </div>
        <div class="output" id="output">
            <div class="output-header">
                <h2>Generated LaTeX Research Paper</h2>
                <button class="copy-btn" id="copy-btn">Copy to Clipboard</button>
            </div>
            <pre id="latex-output"></pre>
            <div class="next-steps">
                <h3>Next Steps:</h3>
                <ol>
                    <li>Copy the LaTeX code to your preferred LaTeX editor (Overleaf, TeXstudio, etc.)</li>
                    <li>Check for any compilation errors and make minor adjustments if needed</li>
                    <li>Compile the document to generate your PDF research paper</li>
                    <li>Review and modify the content as necessary for your specific needs</li>
                    <li>If using Overleaf, create a new project and paste this code to immediately see the compiled PDF</li>
                </ol>
            </div>
        </div>
    </div>
    <footer>
        <p>Powered by OpenRouter AI &copy; 2025</p>
    </footer>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const topicInput = document.getElementById('topic');
            const generateBtn = document.getElementById('generate-btn');
            const loading = document.getElementById('loading');
            const output = document.getElementById('output');
            const latexOutput = document.getElementById('latex-output');
            const copyBtn = document.getElementById('copy-btn');
            const errorMessage = document.getElementById('error-message');
            const stages = document.querySelectorAll('.stage');
            function simulateProgress() {
                const stageDurations = [8000, 15000, 60000, 25000, 30000, 20000];
                let currentStage = 0;
                function updateStage() {
                    if (currentStage < stages.length) {
                        stages.forEach((stage, index) => {
                            stage.classList.remove('active');
                            if (index < currentStage) {
                                stage.classList.add('completed');
                            }
                        });
                        stages[currentStage].classList.add('active');
                        currentStage++;
                        if (currentStage < stages.length) {
                            setTimeout(updateStage, stageDurations[currentStage - 1]);
                        }
                    }
                }
                updateStage();
            }
            generateBtn.addEventListener('click', async function() {
                const topic = topicInput.value.trim();
                if (!topic) {
                    errorMessage.textContent = 'Please enter a research topic or title';
                    errorMessage.style.display = 'block';
                    return;
                }
                errorMessage.style.display = 'none';
                loading.style.display = 'block';
                output.style.display = 'none';
                generateBtn.disabled = true;
                stages.forEach(stage => {
                    stage.classList.remove('active', 'completed');
                });
                simulateProgress();
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ topic })
                    });
                    const data = await response.json();
                    if (data.error) {
                        errorMessage.textContent = data.error;
                        errorMessage.style.display = 'block';
                    } else {
                        stages.forEach(stage => {
                            stage.classList.remove('active');
                            stage.classList.add('completed');
                        });
                        latexOutput.textContent = data.paper;
                        output.style.display = 'block';
                    }
                } catch (error) {
                    errorMessage.textContent = 'An error occurred. Please try again.';
                    errorMessage.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                    generateBtn.disabled = false;
                }
            });
            copyBtn.addEventListener('click', function() {
                navigator.clipboard.writeText(latexOutput.textContent)
                    .then(() => {
                        const originalText = copyBtn.textContent;
                        copyBtn.textContent = 'Copied!';
                        setTimeout(() => {
                            copyBtn.textContent = originalText;
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Failed to copy: ', err);
                    });
            });
        });
    </script>
</body>
</html>
        """)

if __name__ == '__main__':
    create_template_files()
    app.run(debug=True)
