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
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
        if (!html.startsWith('<p>')) html = '<p>' + html + '</p>';
        html = html.replace(/<p><\/p>/g, '');
        return html;
    }

    // Typing cursor element
    function cursor() {
        return '<span class="typing-cursor">▍</span>';
    }

    function setLoading() {
        analyzeBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnIcon.classList.add('hidden');
        spinner.classList.remove('hidden');
        reasoningOutput.innerHTML = '<div class="empty-state"><i class="fa-solid fa-brain fa-fade"></i><p>Generating internal monologue...</p></div>';
        finalAnswerOutput.innerHTML = '<div class="empty-state"><i class="fa-solid fa-spinner fa-spin"></i><p>Awaiting reasoning completion...</p></div>';
        timeBadge.textContent = 'Streaming...';
    }

    function resetButton() {
        analyzeBtn.disabled = false;
        btnText.textContent = 'Generate Clinical Reasoning';
        btnIcon.classList.remove('hidden');
        spinner.classList.add('hidden');
    }

    async function generateReasoning() {
        const question = inputArea.value.trim();
        if (!question) {
            alert('Please enter a clinical scenario first.');
            inputArea.focus();
            return;
        }

        setLoading();

        // Use fetch with streaming to read SSE tokens
        let streamingText = '';
        let reasoningStarted = false;
        let startTime = Date.now();

        try {
            const response = await fetch('/api/reason/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Server error occurred');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            // Show the streaming panel immediately
            reasoningOutput.innerHTML = cursor();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr) continue;

                    let payload;
                    try { payload = JSON.parse(jsonStr); } catch { continue; }

                    if (payload.error) {
                        throw new Error(payload.error);
                    }

                    if (!payload.done) {
                        // Stream individual token into the reasoning panel
                        streamingText += payload.token;
                        // Show raw streamed text with a live cursor
                        reasoningOutput.innerHTML = formatText(streamingText) + cursor();
                        // Auto-scroll to bottom
                        reasoningOutput.scrollTop = reasoningOutput.scrollHeight;
                        
                        // Update live timer
                        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                        timeBadge.textContent = `${elapsed}s`;
                    } else {
                        // Final event — replace with properly parsed sections
                        reasoningOutput.innerHTML = formatText(payload.reasoning_chain || streamingText);
                        
                        // Dramatic reveal of final answer
                        setTimeout(() => {
                            finalAnswerOutput.innerHTML = formatText(payload.final_answer || '');
                            finalAnswerOutput.parentElement.animate([
                                { backgroundColor: 'rgba(16, 185, 129, 0.2)', transform: 'scale(0.98)' },
                                { backgroundColor: 'transparent', transform: 'scale(1)' }
                            ], { duration: 600, easing: 'ease-out' });
                        }, 300);

                        timeBadge.textContent = `${payload.generation_time_s}s`;
                        resetButton();
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            reasoningOutput.innerHTML = `<div style="color: #ef4444;"><i class="fa-solid fa-circle-exclamation"></i> Error: ${error.message}</div>`;
            finalAnswerOutput.innerHTML = `<div class="empty-state"><i class="fa-solid fa-xmark"></i><p>Failed to generate output.</p></div>`;
            timeBadge.textContent = 'Error';
            resetButton();
        }
    }

    analyzeBtn.addEventListener('click', generateReasoning);

    // Ctrl+Enter to submit
    inputArea.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            generateReasoning();
        }
    });
});
