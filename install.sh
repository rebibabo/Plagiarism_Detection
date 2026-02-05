nltkd () {
  python - "$@" <<'EOF'
import ssl
import nltk
import sys

ssl._create_default_https_context = ssl._create_unverified_context

if len(sys.argv) < 2:
    print("Usage: nltkd <package_name>")
    sys.exit(1)

for pkg in sys.argv[1:]:
    nltk.download(pkg)
EOF
}

nltkd punkt
nltkd punkt_tb
nltkd stopwords
nltkd brown
nltkd cmudict
nltkd omw-1.4
nltkd wordnet
nltkd averaged_perceptron_tagger

wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
pip install en_core_web_sm-3.8.0-py3-none-any.whl
rm en_core_web_sm-3.8.0-py3-none-any.whl