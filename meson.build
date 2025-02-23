project('profile_hmm', 'cpp',
        version : '1.0.0',
        default_options : ['warning_level=3', 'cpp_std=c++23'])


argparse_proj = subproject('argparse')
argparse_dep = argparse_proj.get_variable('argparse_dep')


profile_hmm = executable('profile_hmm', 'src/main.cpp', dependencies : [argparse_dep], install : true)

test('test', profile_hmm)