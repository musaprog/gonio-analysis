import os
import sys
# Add the directory above the pupil folder to path because
# other needed packages may lay there
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from pupil.drosom.gui.examine import main

if __name__ == "__main__":
    main()


