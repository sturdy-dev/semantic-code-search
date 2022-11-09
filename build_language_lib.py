from tree_sitter import Language

Language.build_library(
        'build/my-languages.so',
        [
            '/Users/kiril/src/tree-sitter-ruby',
            '/Users/kiril/src/tree-sitter-go',
            '/Users/kiril/src/tree-sitter-rust',
            '/Users/kiril/src/tree-sitter-java',
            '/Users/kiril/src/tree-sitter-vue',
            '/Users/kiril/src/tree-sitter-svelte',
            '/Users/kiril/src/tree-sitter-javascript',
            '/Users/kiril/src/tree-sitter-typescript/typescript',
            '/Users/kiril/src/tree-sitter-python',
        ])

print('saved to build/my-languages.so')