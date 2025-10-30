#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow


# In[2]:


import os
import git
import ast
import logging
import csv
import pandas as pd
import sys # For getting current exception info


# In[4]:


# --- Configuration ---
INPUT_SEART_CSV = "seart_repos.csv" 
OUTPUT_REPO_LIST = "repositories.txt"

REPO_LIST_FILE = "repositories.txt" 
CLONE_DIR = "repos"

# Define a dedicated output data directory 
OUTPUT_DATA_DIR = "collected_data" # All CSVs and description files will go here

# Define your output CSV and description file paths relative to OUTPUT_DATA_DIR
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DATA_DIR, "python_functions.csv")
DESCRIPTION_FILE = os.path.join(OUTPUT_DATA_DIR, "dataset_description.txt")

MAX_FUNCTIONS = 250000 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# In[ ]:


class PythonFunctionExtractor(ast.NodeVisitor):
    """
    An AST visitor that extracts the source code and metadata of Python function definitions.
    """
    def __init__(self, full_source_code): # <-- Renamed parameter for clarity
        self.full_source_code = full_source_code # <-- Store the full string
        self.functions = []
        
    def visit_FunctionDef(self, node):
        """Called for 'def' function definitions."""
        logging.debug(f"DEBUG: Found FunctionDef: {node.name} at line {node.lineno}") # ADD THIS DEBUG LOG
        self._extract_function_info(node)
        self.generic_visit(node) # Continue for nested functions

    def visit_AsyncFunctionDef(self, node):
        """Called for 'async def' function definitions."""
        logging.debug(f"DEBUG: Found AsyncFunctionDef: {node.name} at line {node.lineno}") # ADD THIS DEBUG LOG
        self._extract_function_info(node)
        self.generic_visit(node)

    def _extract_function_info(self, node):
        try:
            # Pass the original full source code string here
            original_code = ast.get_source_segment(source=self.full_source_code, node=node) # <-- CHANGE IS HERE
            if original_code:
                # ast.FunctionDef provides lineno and end_lineno (Python 3.8+)
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line + original_code.count('\n')) 
                
                signature = f"def {node.name}(...)" 
                if isinstance(node, ast.AsyncFunctionDef):
                    signature = f"async {signature}"

                self.functions.append({
                    "function_name": node.name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "signature": signature,
                    "original_code": original_code,
                    "code_tokens": " ".join(original_code.split())
                })
                logging.debug(f"DEBUG: Successfully extracted source for '{node.name}'. Length: {len(original_code)} chars.")
            else:
                logging.debug(f"DEBUG: ast.get_source_segment returned empty for '{node.name}'.")
        except (TypeError, ValueError) as e:
            logging.warning(f"Could not extract source for a function at line {node.lineno}: {e}")
        except Exception:
            logging.error(f"Unexpected error extracting function at line {node.lineno}: {sys.exc_info()[0]}")


# In[ ]:


def clean_functions(function_info, counters):
    """
    Applies cleaning rules to a dictionary representing a function.
    Updates counters for removed functions.
    """
    start, end = function_info.get("start_line"), function_info.get("end_line")
    code = function_info.get("original_code", "")

    if start is None or end is None or (isinstance(start, int) and isinstance(end, int) and start > end):
        counters['invalid_lines'] += 1
        return None
    
    # Consider the actual number of lines of code, excluding leading/trailing whitespace lines
    code_lines = [line.strip() for line in code.splitlines() if line.strip()]
    num_lines = len(code_lines)

    if num_lines < 3: # Minimum 3 lines of *actual* code
        counters['short_functions'] += 1
        return None
    if num_lines > 100: # Maximum 100 lines of *actual* code
        counters['long_functions'] += 1
        return None
    if not code.strip(): # Check if code is empty after stripping whitespace
        counters['empty_functions'] += 1
        return None
    
    # Optional: Filter out functions with very high comment-to-code ratio if desired
    # For now, we'll keep them as comments are part of code context.

    return function_info


# In[ ]:


def clone_repositories():
    """
    Clones repositories from the list specified in REPO_LIST_FILE.
    """
    if not os.path.exists(REPO_LIST_FILE):
        logging.error(f"'{REPO_LIST_FILE}' not found. Please create it and add repository URLs.")
        return False

    if not os.path.exists(CLONE_DIR):
        logging.info(f"Creating directory for repositories: '{CLONE_DIR}'")
        os.makedirs(CLONE_DIR)

    repos_to_process = []
    with open(REPO_LIST_FILE, 'r') as f:
        for line in f:
            repo_url = line.strip()
            if repo_url and not repo_url.startswith('#'):
                repos_to_process.append(repo_url)
    
    if not repos_to_process:
        logging.warning(f"No repository URLs found in '{REPO_LIST_FILE}'. Please add some.")
        return False

    for repo_url in repos_to_process:
        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            local_path = os.path.join(CLONE_DIR, repo_name)

            if os.path.exists(local_path):
                logging.info(f"Repository '{repo_name}' already exists. Skipping clone.")
            else:
                logging.info(f"Cloning '{repo_url}' into '{local_path}'...")
                git.Repo.clone_from(repo_url, local_path)
                logging.info(f"Successfully cloned '{repo_name}'.")
        except git.exc.GitCommandError as e:
            logging.error(f"Failed to clone {repo_url}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred for {repo_url}: {e}")
    return True


# In[ ]:


def process_cloned_repositories():
    """
    Walks through cloned repos, extracts, cleans Python functions, and writes to CSV.
    """
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    
    # Initialize counters for cleaning statistics
    counters = {
        'total_scanned_files': 0,
        'total_parsed_files': 0,
        'total_skipped_files_syntax_error': 0,
        'total_skipped_files_other_error': 0,
        'total_extracted_functions_raw': 0,
        'short_functions': 0,
        'long_functions': 0,
        'empty_functions': 0,
        'invalid_lines': 0,
        'total_written_functions_cleaned': 0
    }

    # Set up CSV writer, managing header creation
    csv_file_exists = os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0
    fieldnames = [
        "repo_name", "repo_url", "file_path", "function_name", "start_line",
        "end_line", "signature", "original_code", "code_tokens"
    ]
    
    f_csv = open(OUTPUT_CSV_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    if not csv_file_exists:
        writer.writeheader()

    seen_function_hashes = set() # To store hashes for deduplication

    logging.info(f"Starting function extraction and writing to '{OUTPUT_CSV_FILE}'...")

    for root, _, files in os.walk(CLONE_DIR):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # Determine repo_name and repo_url from the file_path
                relative_path = os.path.relpath(file_path, CLONE_DIR)
                repo_name_guess = relative_path.split(os.sep)[0]
                # A more robust way might be to pass repo_url from clone_repositories
                # For simplicity here, we'll try to reconstruct.
                repo_url_guess = "https://github.com/unknown/unknown" # Placeholder, improve if needed

                # Get the actual repo URL from REPO_LIST_FILE based on repo_name_guess
                with open(REPO_LIST_FILE, 'r') as rlf:
                    for line in rlf:
                        if repo_name_guess in line:
                            repo_url_guess = line.strip()
                            break


                counters['total_scanned_files'] += 1
                if counters['total_scanned_files'] % 1000 == 0:
                    logging.info(f"Scanned {counters['total_scanned_files']} Python files...")
                    logging.info(f"Currently extracted and cleaned: {counters['total_written_functions_cleaned']} functions.")
                
                if counters['total_written_functions_cleaned'] >= MAX_FUNCTIONS:
                    logging.info(f"Reached MAX_FUNCTIONS ({MAX_FUNCTIONS}). Stopping collection.")
                    f_csv.close()
                    return counters # Early exit

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()
                    
                    tree = ast.parse(file_content)
                    counters['total_parsed_files'] += 1
                    
                    extractor = PythonFunctionExtractor(file_content)
                    extractor.visit(tree)

                    for func_data in extractor.functions:
                        counters['total_extracted_functions_raw'] += 1
                        
                        # Add repo and file path metadata
                        full_func_info = {
                            "repo_name": repo_name_guess,
                            "repo_url": repo_url_guess,
                            "file_path": os.path.relpath(file_path, os.path.join(CLONE_DIR, repo_name_guess)),
                            **func_data
                        }

                        cleaned_func = clean_functions(full_func_info, counters)

                        if cleaned_func:
                            # Deduplicate based on a hash of the cleaned code content
                            # This catches exact duplicates across files/repos
                            func_hash = hash(cleaned_func['original_code'])
                            if func_hash not in seen_function_hashes:
                                writer.writerow(cleaned_func)
                                seen_function_hashes.add(func_hash)
                                counters['total_written_functions_cleaned'] += 1
                            else:
                                # Not explicitly counted, but implicitly removed by set check
                                pass 
                            
                            if counters['total_written_functions_cleaned'] >= MAX_FUNCTIONS:
                                logging.info(f"Reached MAX_FUNCTIONS ({MAX_FUNCTIONS}). Stopping collection.")
                                f_csv.close()
                                return counters # Early exit

                except SyntaxError:
                    logging.warning(f"Skipping {file_path} due to SyntaxError.")
                    counters['total_skipped_files_syntax_error'] += 1
                except Exception as e:
                    logging.error(f"An unexpected error occurred processing {file_path}: {e}")
                    counters['total_skipped_files_other_error'] += 1
    
    f_csv.close() # Ensure CSV file is closed at the end of successful run
    logging.info("Function extraction and writing to CSV complete.")
    return counters


# In[ ]:


def generate_description_file(counters):
    """
    Creates a summary description file for the collected dataset.
    """
    os.makedirs(os.path.dirname(DESCRIPTION_FILE), exist_ok=True)
    
    description_text = f"""
    Dataset Collection Summary (Python Functions):

    Configuration:
    - Repository List: {REPO_LIST_FILE}
    - Clone Directory: {CLONE_DIR}
    - Output CSV: {OUTPUT_CSV_FILE}
    - Maximum Functions Target: {MAX_FUNCTIONS}

    Processing Statistics:
    - Total Python files scanned: {counters['total_scanned_files']}
    - Total Python files successfully parsed (AST): {counters['total_parsed_files']}
    - Files skipped due to SyntaxError: {counters['total_skipped_files_syntax_error']}
    - Files skipped due to other errors: {counters['total_skipped_files_other_error']}

    Function Extraction Statistics (Raw):
    - Total functions initially extracted (before cleaning): {counters['total_extracted_functions_raw']}

    Function Cleaning Statistics:
    - Functions removed:
        - Too short (<3 lines of actual code): {counters['short_functions']}
        - Too long (>100 lines of actual code): {counters['long_functions']}
        - Empty (after stripping whitespace): {counters['empty_functions']}
        - Invalid line numbers/structure: {counters['invalid_lines']}
        - Duplicates (based on code hash): {counters['total_extracted_functions_raw'] - counters['total_written_functions_cleaned'] - (counters['short_functions'] + counters['long_functions'] + counters['empty_functions'] + counters['invalid_lines'])}
          (This is an estimate of only *deduplicated* functions that passed other filters. The true duplicate count is more complex to show here.)
          
    Final Dataset:
    - Total unique, cleaned functions written to CSV: {counters['total_written_functions_cleaned']}
    
    Each row in '{OUTPUT_CSV_FILE}' represents a cleaned Python function and includes:
    repo_name, repo_url, file_path, function_name, start_line, end_line, signature, original_code, code_tokens (simple split).
    """

    with open(DESCRIPTION_FILE, "w", encoding="utf-8") as f:
        f.write(description_text)
    logging.info(f"Dataset description saved to '{DESCRIPTION_FILE}'.")


# In[ ]:


def generate_repo_list_from_seart_csv(input_csv, output_txt):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    try:
        # Read the CSV. The 'name' column appears to be in the format 'owner/repo'
        df = pd.read_csv(input_csv)
        
        # Ensure the 'name' column exists
        if 'name' not in df.columns:
            print(f"Error: '{input_csv}' does not contain a 'name' column.")
            return

        # Generate GitHub URLs
        github_urls = [f"https://github.com/{repo_name}" for repo_name in df['name'].unique()]

        with open(output_txt, 'w') as f:
            for url in github_urls:
                f.write(url + "\n")
        print(f"Successfully generated '{output_txt}' with {len(github_urls)} URLs.")
    except Exception as e:
        print(f"An error occurred: {e}")


# In[ ]:





# In[ ]:


#Get the code and repo

if __name__ == "__main__":
    
    # 1. Ensure the main output data directory exists
    if not os.path.exists(OUTPUT_DATA_DIR):
        logging.info(f"Creating output data directory: '{OUTPUT_DATA_DIR}'")
        os.makedirs(OUTPUT_DATA_DIR)

    # 2. Clone repositories
    if clone_repositories():
        # 3. Process cloned repositories, extract functions, clean, and write to CSV
        final_counters = process_cloned_repositories()
        
        # 4. Generate a description file based on collected statistics
        generate_description_file(final_counters)
    else:
        logging.error("Repository cloning step failed. Please check your 'repositories.txt' and internet connection.")

