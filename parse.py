#!/usr/bin/python

import sys

from grammar import Grammar

class Parser:

	def __init__(self, grammar_file):
		self.grammar = Grammar(grammar_file)

def main():

	if len(sys.argv) != 3:
		return

	file_gr = sys.argv[1]
	file_sen = sys.argv[2]

	parser = Parser(file_gr)

if __name__ == "__main__":
	main()
