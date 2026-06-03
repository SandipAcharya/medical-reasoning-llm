document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const inputArea = document.getElementById('patient-input');
    const reasoningOutput = document.getElementById('reasoning-output');
    const finalAnswerOutput = document.getElementById('final-answer-output');
    const timeBadge = document.getElementById('time-badge');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnIcon = analyzeBtn.querySelector('.btn-icon');
    const spinner = analyzeBtn.querySelector('.spinner');

    // Basic Markdown to HTML converter
    function formatText(text) {
        if (!text) return '';
        
        let html = text
            // Replace bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Replace line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
            
        // Wrap in p tags if not already
        if (!html.startsWith('<p>')) {
            html = '<p>' + html + '</p>';
        }
        
        // Fix empty paragraphs
        html = html.replace(/<p><\/p>/g, '');
        
        return html;
    }

    async function generateReasoning() {
        const question = inputArea.value.trim();
        
        if (!question) {
            alert('Please enter a clinical scenario first.');
            inputArea.focus();
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnIcon.classList.add('hidden');
        spinner.classList.remove('hidden');
        
        reasoningOutput.innerHTML = '<div class="empty-state"><i class="fa-solid fa-brain fa-fade"></i><p>Generating internal monologue...</p></div>';
        finalAnswerOutput.innerHTML = '<div class="empty-state"><i class="fa-solid fa-spinner fa-spin"></i><p>Awaiting reasoning completion...</p></div>';
        timeBadge.textContent = 'Running...';

        try {
            const response = await fetch('/api/reason', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            // Populate Data
            reasoningOutput.innerHTML = formatText(data.reasoning_chain);
            
            // Add a subtle delay before showing the final answer for dramatic effect
            setTimeout(() => {
                finalAnswerOutput.innerHTML = formatText(data.final_answer);
                finalAnswerOutput.parentElement.animate([
                    { backgroundColor: 'rgba(16, 185, 129, 0.2)', transform: 'scale(0.98)' },
                    { backgroundColor: 'transparent', transform: 'scale(1)' }
                ], { duration: 500, easing: 'ease-out' });
            }, 500);

            timeBadge.textContent = `${data.generation_time_s}s`;

        } catch (error) {
            console.error('Error:', error);
            reasoningOutput.innerHTML = `<div style="color: #ef4444;"><i class="fa-solid fa-circle-exclamation"></i> Error: ${error.message}</div>`;
            finalAnswerOutput.innerHTML = `<div class="empty-state"><i class="fa-solid fa-xmark"></i><p>Failed to generate output.</p></div>`;
            timeBadge.textContent = 'Error';
        } finally {
            // Restore Button State
            analyzeBtn.disabled = false;
            btnText.textContent = 'Generate Clinical Reasoning';
            btnIcon.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }

    analyzeBtn.addEventListener('click', generateReasoning);
    
    // Command + Enter to submit
    inputArea.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            generateReasoning();
        }
    });
});
