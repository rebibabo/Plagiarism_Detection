document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('inputText');
    const detectBtn = document.getElementById('detectBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const statsInfo = document.getElementById('statsInfo');
    const sentenceCount = document.getElementById('sentenceCount');

    // Detect button click handler
    detectBtn.addEventListener('click', async function() {
        const text = inputText.value.trim();

        if (!text) {
            showError('Please enter text content');
            return;
        }

        // Disable button and show loading
        detectBtn.disabled = true;
        const btnText = detectBtn.querySelector('.btn-text');
        const spinner = detectBtn.querySelector('.loading-spinner');
        btnText.style.display = 'none';
        spinner.style.display = 'inline-block';

        try {
            const response = await fetch('/api/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    threshold: 0.5
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Detection failed');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            showError('Error: ' + error.message);
            console.error('Error:', error);
        } finally {
            // Re-enable button and hide loading
            detectBtn.disabled = false;
            btnText.style.display = 'inline';
            spinner.style.display = 'none';
        }
    });

    function displayResults(data) {
        const results = data.results;
        const numSentences = data.num_sentences;

        if (numSentences === 0) {
            resultsContainer.innerHTML = '<p class="placeholder">No valid sentences detected</p>';
            statsInfo.style.display = 'none';
            updateSeverityLabel('-', 'No Result');
            return;
        }

        // Show stats
        statsInfo.style.display = 'block';
        sentenceCount.textContent = numSentences;

        // Clear previous results
        resultsContainer.innerHTML = '';

        // Calculate average probability and determine severity
        let totalProb = 0;
        results.forEach(result => {
            totalProb += result.plagiarism_prob;
        });
        const avgProb = totalProb / numSentences;

        // Display each sentence with color coding
        results.forEach((result, index) => {
            const prob = result.plagiarism_prob;
            let colorClass = 'low';

            if (prob >= 0.9) {
                colorClass = 'high';
            } else if (prob >= 0.8) {
                colorClass = 'mid-high';
            } else if (prob >= 0.7) {
                colorClass = 'mid';
            } else if (prob >= 0.6) {
                colorClass = 'medium';
            } else if (prob >= 0.5) {
                colorClass = 'low-mid';
            }

            const sentenceEl = document.createElement('div');
            sentenceEl.className = `sentence-item ${colorClass}`;
            sentenceEl.setAttribute('data-sentence-id', index);
            sentenceEl.setAttribute('data-offset-start', result.offset_start);
            sentenceEl.setAttribute('data-offset-end', result.offset_end);
            sentenceEl.innerHTML = `
                <span class="sentence-text">${escapeHtml(result.sentence)}</span>
                <div class="probability-badge">
                    Probability: ${(prob * 100).toFixed(1)}%
                    <span class="tooltip">Plagiarism Probability: ${(prob * 100).toFixed(2)}%</span>
                </div>
            `;

            // Add click event listener to highlight in left panel
            sentenceEl.addEventListener('click', function() {
                highlightSentenceInTextarea(result.offset_start, result.offset_end);
            });

            resultsContainer.appendChild(sentenceEl);
        });

        // Update severity label based on average
        updateSeverityLabel((avgProb * 100).toFixed(1) + '%', getSeverityLabel(avgProb));
    }

    function getSeverityLabel(prob) {
        if (prob >= 0.9) {
            return 'High Suspicious';
        } else if (prob >= 0.8) {
            return 'Mid-High Suspicious';
        } else if (prob >= 0.7) {
            return 'Mid Suspicious';
        } else if (prob >= 0.6) {
            return 'Low-Mid Suspicious';
        } else if (prob >= 0.5) {
            return 'Low Suspicious';
        } else {
            return 'Low Probability';
        }
    }

    function highlightSentenceInTextarea(startOffset, endOffset) {
        // Focus the textarea
        inputText.focus();
        
        // Set selection to the sentence
        inputText.setSelectionRange(startOffset, endOffset);
        
        // Scroll to the selected text
        const textarea = inputText;
        const scrollPosition = (startOffset / textarea.value.length) * (textarea.scrollHeight - textarea.clientHeight);
        textarea.scrollTop = scrollPosition;
        
        // Add highlight class temporarily
        inputText.classList.add('highlight');
        
        // Remove highlight after 2 seconds
        setTimeout(() => {
            inputText.classList.remove('highlight');
        }, 2000);
    }

    function updateSeverityLabel(prob, label) {
        const labelItem = document.querySelector('.label-item');
        if (labelItem) {
            labelItem.innerHTML = `
                <span class="label-text">${label}</span>
                <span class="label-prob">${prob}</span>
            `;
        }
    }

    function showError(message) {
        resultsContainer.innerHTML = `<div class="error-message">${escapeHtml(message)}</div>`;
        statsInfo.style.display = 'none';
        updateSeverityLabel('-', 'Error');
    }

    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    // Allow Enter key to trigger detection (Ctrl+Enter or Cmd+Enter)
    inputText.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            detectBtn.click();
        }
    });
});
