#!/usr/bin/env bash
# Makefile-like script for common development tasks
# Usage: ./dev.sh <command>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
function print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Commands
case "${1:-help}" in
    help)
        print_header "Available Commands"
        echo "  ./dev.sh setup         - Install dependencies and setup"
        echo "  ./dev.sh install       - Install Python dependencies"
        echo "  ./dev.sh run           - Run the API server locally"
        echo "  ./dev.sh test          - Run API tests"
        echo "  ./dev.sh docker-build  - Build Docker image"
        echo "  ./dev.sh docker-run    - Run with Docker"
        echo "  ./dev.sh docker-stop   - Stop Docker containers"
        echo "  ./dev.sh compose-up    - Start with docker-compose"
        echo "  ./dev.sh compose-down  - Stop docker-compose services"
        echo "  ./dev.sh logs          - Show docker-compose logs"
        echo "  ./dev.sh format        - Format code with black"
        echo "  ./dev.sh lint          - Run linting checks"
        ;;
    
    setup)
        print_header "Setting up project"
        python setup.py
        print_success "Setup complete"
        ;;
    
    install)
        print_header "Installing dependencies"
        pip install -r requirements.txt
        print_success "Dependencies installed"
        ;;
    
    run)
        print_header "Starting API server"
        print_info "API will be available at http://localhost:8000"
        print_info "Documentation at http://localhost:8000/docs"
        python main.py
        ;;
    
    test)
        print_header "Running tests"
        if [ -n "$2" ]; then
            python test_api.py --image "$2"
        else
            python test_api.py
        fi
        ;;
    
    docker-build)
        print_header "Building Docker image"
        docker build -t crop-model:latest .
        print_success "Docker image built"
        ;;
    
    docker-run)
        print_header "Running with Docker"
        docker run -p 8000:8000 \
            -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
            -v "$(pwd)/processed_data:/app/processed_data:ro" \
            crop-model:latest
        ;;
    
    docker-stop)
        print_header "Stopping Docker containers"
        docker ps -q | xargs -r docker stop
        print_success "Containers stopped"
        ;;
    
    compose-up)
        print_header "Starting with docker-compose"
        docker-compose up -d
        print_success "Services started"
        docker-compose ps
        ;;
    
    compose-down)
        print_header "Stopping docker-compose services"
        docker-compose down
        print_success "Services stopped"
        ;;
    
    logs)
        print_header "Docker compose logs"
        docker-compose logs -f api
        ;;
    
    format)
        print_header "Formatting code"
        which black > /dev/null && {
            black main.py test_api.py setup.py client_examples.py
            print_success "Code formatted"
        } || {
            print_error "black not installed (pip install black)"
        }
        ;;
    
    lint)
        print_header "Running linting checks"
        which pylint > /dev/null && {
            pylint main.py test_api.py setup.py || true
            print_success "Linting complete"
        } || {
            print_info "pylint not installed (pip install pylint)"
        }
        ;;
    
    *)
        print_error "Unknown command: $1"
        echo "Run './dev.sh help' for available commands"
        exit 1
        ;;
esac
