import subprocess
import os
import time

def main():
    # Define the exact modes you want to run
    modes = ["no_sst", "no_dhw", "time_only", "no_graph"]
    
    # Define where to save the final text file
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ablation_summary.txt")
    
    print("=== STARTING AUTOMATED ABLATION BATCH ===")
    print(f"Results will be appended to: {output_file}\n")
    
    # Initialize/Clear the log file with a header
    with open(output_file, "w") as f:
        f.write("=======================================================\n")
        f.write(" MASTER ABLATION STUDY RESULTS LOG \n")
        f.write("=======================================================\n\n")

    total_start_time = time.time()

    # Loop through each mode and run the command
    for mode in modes:
        print(f"[{time.strftime('%H:%M:%S')}] Executing: python src/ablation_study.py --mode {mode}")
        
        # This executes the command exactly as if you typed it in the terminal
        result = subprocess.run(
            ["python", "src/ablation_study.py", "--mode", mode],
            capture_output=True, # Captures the print statements
            text=True            # Keeps it as a readable string
        )
        
        # Save the specific output to the text file
        with open(output_file, "a") as f:
            f.write(f"\n\n{'='*55}\n")
            f.write(f" RUN LOG FOR MODE: {mode.upper()}\n")
            f.write(f"{'='*55}\n")
            f.write(result.stdout) # Writes the normal print statements
            
            # If the script crashed for any reason, log the error trace too
            if result.stderr:
                f.write("\n[ERRORS RECORDED]:\n")
                f.write(result.stderr)
                
        print(f"[{time.strftime('%H:%M:%S')}] Finished {mode.upper()}. Output saved.\n")

    total_time = (time.time() - total_start_time) / 60
    print("=======================================================")
    print(f"ALL RUNS COMPLETE! Total time: {total_time:.1f} minutes.")
    print(f"Open '{output_file}' to copy your numbers for the manuscript.")
    print("=======================================================")

if __name__ == "__main__":
    main()