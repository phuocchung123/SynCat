# SynCat

ReCatAI is a cutting-edge repository dedicated to harnessing the power of artificial intelligence to categorize chemical reactions, particularly within the realm of organic chemistry. This project aims to bridge the gap between traditional chemical analysis and modern computational methodologies, offering a novel approach to understanding and organizing chemical reactions.

![screenshot](./Image/fig_repo.webp)


## Step-by-Step Installation Guide

1. **Python Installation:**
  Ensure that Python 3.11 or later is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Creating a Virtual Environment (Optional but Recommended):**
  It's recommended to use a virtual environment to avoid conflicts with other projects or system-wide packages. Use the following commands to create and activate a virtual environment:

  ```bash
  python -m venv syncat-env
  source syncat-env/bin/activate  
  ```
  Or Conda

  ```bash
  conda create --name syncat-env python=3.11
  conda activate syncat-env
  ```

3. **Cloning and Installing SynCat:**
  Clone the SynCat repository from GitHub and install it:

  ```bash
  git clone https://github.com/phuocchung123/SynCat.git
  cd SynCat
  pip install -r requirements.txt
  pip install black flake8 pytest # black for formating, flake8 for checking format, pytest for testing
  ```


## Setting Up Your Development Environment

Before you start, ensure your local development environment is set up correctly. Pull the latest version of the `main` branch to start with the most recent stable code.

```bash
git checkout main
git pull
```

## Working on New Features

1. **Create a New Branch**:  
   For every new feature or bug fix, create a new branch from the `main` branch. Name your branch meaningfully, related to the feature or fix you are working on.

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Commit Changes**:  
   Make your changes locally, commit them to your branch. Keep your commits small and focused; each should represent a logical unit of work.

   ```bash
   git commit -m "Describe the change"
   ```

3. **Run Quality Checks**:  
   Before finalizing your feature, run the following commands to ensure your code meets our formatting standards and passes all tests:

   ```bash
   ./lint.sh # Check code format
   pytest Test # Run tests
   ```

   Fix any issues or errors highlighted by these checks.

## Integrating Changes

1. **Rebase onto Staging**:  
   Once your feature is complete and tests pass, rebase your changes onto the `staging` branch to prepare for integration.

   ```bash
   git fetch origin
   git rebase origin/staging
   ```

   Carefully resolve any conflicts that arise during the rebase.

2. **Push to Your Feature Branch**:
   After successfully rebasing, push your branch to the remote repository.

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   Open a pull request from your feature branch to the `staging` branch. Ensure the pull request description clearly describes the changes and any additional context necessary for review.

## Important Notes