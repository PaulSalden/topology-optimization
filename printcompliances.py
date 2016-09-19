import os, sys, pickle
for root, dirs, files in os.walk(sys.argv[1]):
    for file in sorted(files):
        if file.endswith(".p"):
             p = pickle.load(open(os.path.join(root, file), "rb"))

             try:
                if len(sys.argv) >= 3 and sys.argv[2] == "s":
                  string = "  {} & {} & \\num{{{:01.3e}}} & {} \\\\"
                else:
                  string = "  {} & {} & {:01.3f} & {} \\\\"
                
                it = "{}*".format(p["n_it"]) if p["n_it"] == 999 else p["n_it"]
                time = str(p["time"]).split(".")[0]
 
                print string.format(file, it, p["compliances"][-1], time)

             except Exception:
               print "problem!"
