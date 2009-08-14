#!/usr/bin/env python

import unittest
import default_opts
import output
import sys
import StringIO
from mrs import param
from mrs.impl import Serial
from specex import SpecExPSO


class TestSpecEx(unittest.TestCase):

    def setUp(self):
        self.opts = default_opts.default_specex_opts()

    def test_sepso_map(self):
        specex = SpecExPSO(self.opts, [])
        rand = specex.initialization_rand(0)
        particles = list(specex.topology.newparticles(0, rand))
        key = str(particles[2].id)
        value = repr(particles[2])
        emitted_messages = list(specex.sepso_map(key, value))
        specex.just_evaluate(particles[2])
        message = particles[2].make_message_particle()
        expected_messages = [('2', repr(particles[2])),
                ('0', repr(message)),
                ('1', repr(message)),
                ('2', repr(message)),
                ('3', repr(message)),
                ('4', repr(message))]
        for expected_message in expected_messages:
            self.assert_(expected_message in emitted_messages)

    def test_sepso_reduce(self):
        opts = self.opts
        opts.top = 'topology.DRing'
        opts.top__neighbors = 1
        opts.top__num = 10
        opts.top__noselflink = True
        opts.func__dims = 1
        specex = SpecExPSO(self.opts, [])
        key = '0'
        value_iter = ['mp:7;0;1;15.883955477773782;-72.011378817328151;252.30004161989973;15.883955477773782;252.30004161989973;15.883955477773782;',
        'mp:8;0;1;30.481880635459589;56.257684094780927;929.14504707440631;30.481880635459589;929.14504707440631;30.481880635459589;',
        'mp:9;0;1;-42.752068209100081;73.768055162775312;1827.7393361555457;-42.752068209100081;1827.7393361555457;-42.752068209100081;',
        'p:0;0;1;47.764627438385048;-25.777880710367839;2281.4596343277258;47.764627438385048;2281.4596343277258;47.764627438385048;',
        'semp:8;0;2;61.816473973169757;31.334593337710167;3821.2764544755737;61.816473973169757;3821.2764544755737;15.883955477773782;;False;7',
        'semp:8;0;2;61.816473973169757;31.334593337710167;3821.2764544755737;61.816473973169757;3821.2764544755737;15.883955477773782;;True;7',
        'semp:8;0;2;71.541201906522957;41.059321271063361;5118.1435702298841;71.541201906522957;5118.1435702298841;30.481880635459589;;False;-1',
        'semp:8;0;2;71.541201906522957;41.059321271063361;5118.1435702298841;71.541201906522957;5118.1435702298841;30.481880635459589;;True;-1',
        'semp:9;0;2;11.087088613761502;53.839156822861582;122.92353392939994;11.087088613761502;122.92353392939994;-42.752068209100081;;False;-1',
        'semp:9;0;2;11.087088613761502;53.839156822861582;122.92353392939994;11.087088613761502;122.92353392939994;-42.752068209100081;;True;-1',
        'semp:9;0;2;46.38737206373191;89.13944027283199;2151.7882869790956;46.38737206373191;2151.7882869790956;30.481880635459589;;False;8',
        'semp:9;0;2;46.38737206373191;89.13944027283199;2151.7882869790956;46.38737206373191;2151.7882869790956;30.481880635459589;;True;8',
        'sep:0;0;2;10.139376734253396;-37.625250704131652;102.80696055911906;10.139376734253396;102.80696055911906;-42.752068209100081;;False;9',
        'sep:0;0;2;10.139376734253396;-37.625250704131652;102.80696055911906;10.139376734253396;102.80696055911906;-42.752068209100081;;True;9',
        'sep:0;0;2;28.950801330809266;-18.813826107575782;838.14889769598756;28.950801330809266;838.14889769598756;47.764627438385048;;False;-1',
        'sep:0;0;2;28.950801330809266;-18.813826107575782;838.14889769598756;28.950801330809266;838.14889769598756;47.764627438385048;;True;-1']
        expected_output = ['0^p:0;0;3;-93.567063360236261;-103.70644009448966;102.80696055911906;10.139376734253396;102.80696055911906;-42.752068209100081;1827.7393361555457',
        '10^sep:0;0;4;-27.507893111121206;66.059170249115056;102.80696055911906;10.139376734253396;102.80696055911906;-42.752068209100081;1827.7393361555457;False;-1',
        '20^sep:0;0;4;97.243202323812767;190.81026568404903;102.80696055911906;10.139376734253396;102.80696055911906;104.20869385944874;1827.7393361555457;False;9',
        '30^sep:0;0;4;-126.12106202004642;-32.553998659810169;102.80696055911906;-93.567063360236261;102.80696055911906;-42.752068209100081;1827.7393361555457;True;-1',
        '40^sep:0;0;4;-1.3699665851124792;92.197096775123782;102.80696055911906;-93.567063360236261;102.80696055911906;104.20869385944874;1827.7393361555457;True;9']
        actual_output = list(specex.sepso_reduce(key, value_iter))
        self.assertEquals(expected_output, actual_output)

    def test_whole_algorithm(self):
        opts = self.opts
        opts.top = 'topology.DRing'
        opts.top__neighbors = 1
        opts.top__num = 8
        opts.top__noselflink = True
        opts.func__dims = 1
        opts.iters = 10
        opts.out = 'output.Pair'
        save_out = sys.stdout
        sys.stdout = StringIO.StringIO()
        mrs_impl = param.instantiate(opts, 'mrs')
        mrs_impl.program_class = SpecExPSO
        mrs_impl.main(opts, [])
        test_out = sys.stdout
        sys.stdout = save_out
        output = test_out.getvalue()
        output = [line for line in output.splitlines()
                if not (line.startswith('#') or len(line) == 0)]
        expected_output = ['2 0.00116340815465', '4 0.00116340815465',
                '6 0.00116340815465', '8 0.00116340815465',
                '10 0.00116340815465', '12 0.00116340815465',
                '14 0.00116340815465', '16 0.00116340815465',
                '18 0.00116340815465']
        self.assertEquals(expected_output, output)

    def test_whole_algorithm_2(self):
        opts = self.opts
        opts.top = 'topology.Rand'
        opts.top__neighbors = 2
        opts.top__num = 8
        opts.func__dims = 10
        opts.iters = 10
        opts.out = 'output.Pair'
        save_out = sys.stdout
        sys.stdout = StringIO.StringIO()
        mrs_impl = param.instantiate(opts, 'mrs')
        mrs_impl.program_class = SpecExPSO
        mrs_impl.main(opts, [])
        test_out = sys.stdout
        sys.stdout = save_out
        output = test_out.getvalue()
        output = [line for line in output.splitlines()
                if not (line.startswith('#') or len(line) == 0)]
        expected_output = ['2 5412.14094362', '4 4669.31920619',
                '6 4669.31920619', '8 4669.31920619', '10 4669.31920619',
                '12 4468.72038682', '14 2469.19827321', '16 2469.19827321',
                '18 2469.19827321']
        self.assertEquals(expected_output, output)

    def test_whole_algorithm_3(self):
        opts = self.opts
        opts.top = 'topology.Ring'
        opts.top__neighbors = 1
        opts.top__num = 5
        opts.func__dims = 5
        opts.iters = 11
        opts.out = 'output.Pair'
        save_out = sys.stdout
        sys.stdout = StringIO.StringIO()
        mrs_impl = param.instantiate(opts, 'mrs')
        mrs_impl.program_class = SpecExPSO
        mrs_impl.main(opts, [])
        test_out = sys.stdout
        sys.stdout = save_out
        output = test_out.getvalue()
        output = [line for line in output.splitlines()
                if not (line.startswith('#') or len(line) == 0)]
        expected_output = ['2 1572.15761997', '4 1572.15761997',
                '6 1294.37665699', '8 200.329491766', '10 200.329491766',
                '12 200.329491766', '14 200.329491766', '16 200.329491766',
                '18 200.329491766', '20 65.0565230471']
        self.assertEquals(expected_output, output)


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestSpecEx)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

# vim: et sw=4 sts=4
