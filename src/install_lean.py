import subprocess
import platform
import os

def main():
    try:
        # Improved OS detection using platform.system()
        os_name = platform.system()
        print(f"Detected operating system: {os_name}")
        if os_name == "Linux":
            command = "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain stable && . ~/.profile"
        elif os_name == "Darwin":
            command = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/leanprover-community/mathlib4/master/scripts/install_macos.sh)" && source ~/.profile'
        elif os_name == "Windows":
            command = "curl -O --location https://raw.githubusercontent.com/leanprover/elan/master/elan-init.ps1" \
                      " && powershell -ExecutionPolicy Bypass -f elan-init.ps1" \
                      " && del elan-init.ps1"
        else:
            raise RuntimeError("Unsupported operating system. Please install elan manually: https://leanprover-community.github.io/get_started.html")
        subprocess.run(command, shell=True, check=True)
        if os_name in ["Linux", "Darwin"]:
            # add `export PATH="$HOME/.elan/bin:$PATH"` to ~/.bashrc and ~/.zshrc if they exist
            paths = ["~/.bashrc", "~/.zshrc"]
            for path in paths:
                path = os.path.expanduser(path)
                try:
                    with open(path, "a") as file:
                        file.write('\nexport PATH="$HOME/.elan/bin:$PATH"\n')
                    print(f"Added `elan` to {path}")
                except FileNotFoundError:
                    print(f"{path} not found, skipping.")
            print("Please restart your terminal or run 'source ~/.profile' to update your shell with the new environment variables.")
        print("Lean installation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during Lean installation: {e}. Please check https://leanprover-community.github.io/get_started.html for more information.")


if __name__ == "__main__":
    main()
