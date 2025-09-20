#!/usr/bin/env python3
"""
Main entry point for Google App Engine deployment
This imports and runs the Flask app from web_api_wrapper.py
"""

import os
from web_api_wrapper import app

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

