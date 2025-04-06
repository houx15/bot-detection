def log_error(e):
    with open("error.log", "a") as f:
        f.write(str(e) + "\n")
