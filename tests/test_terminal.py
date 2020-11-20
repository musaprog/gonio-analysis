import os
import unittest

import pupilanalysis.drosom.analyser_commands as ac
import pupilanalysis.drosom.terminal as terminal

TESTPATH = os.path.dirname(__file__)

datadir = os.path.join(TESTPATH, 'test_data')
testspecimen = 'test_specimen_01'

class TestTerminal(unittest.TestCase):
    
    def test_analysis_targets(self):
        '''
        Test all the analysis targets in the terminal.py, ie essentially
    
        terminal.py -D datadir -S testspecimen analysis1
        terminal.py -D datadir -S testspecimen analysis2
        ...

        only checking if they can be run successfully without errors.
        '''

        targets = ac.ANALYSER_CMDS

        for target in targets:
            with self.subTest(target=target):
                args = ['--unittest','-D', datadir, '-S', testspecimen, target]
                terminal.main(custom_args=args)
