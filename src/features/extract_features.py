import re
from urllib.parse import urlparse
import tldextract

COMMON_BRANDS = ['bankofamerica', 'hdfc', 'chase', 'paypal', 'wellsfargo']

def extract_features(url):
    features = {}
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    
    # Structural features
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_slash'] = url.count('/')
    features['num_question'] = url.count('?')
    features['num_equal'] = url.count('=')
    features['num_at'] = url.count('@')
    
    # Suspicious patterns
    features['has_script'] = int(bool(re.search(r'<script>|javascript:', url, re.I)))
    features['has_sql_keywords'] = int(bool(re.search(r'union|select|drop|insert|--|xp_cmdshell', url, re.I)))
    features['has_ddos_patterns'] = int(bool(re.search(r'flood|ping|botnet|c2|ddos', url, re.I)))
    
    # Phishing detection
    features['brand_in_subdomain'] = int(any(brand in ext.subdomain.lower() for brand in COMMON_BRANDS))
    features['domain_mismatch'] = int(
        any(brand in ext.subdomain.lower() for brand in COMMON_BRANDS) and 
        not any(brand in ext.domain.lower() for brand in COMMON_BRANDS)
    )
    features['suspicious_tld'] = int(ext.suffix not in ['.com', '.net', '.org'])
    
    return features