const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const solutionPath = (filename) => path.join(__dirname, '../../solutions/css', filename);
const readSolution = (filename) => fs.readFileSync(solutionPath(filename), 'utf8');
const solutionUrl = (filename) => pathToFileURL(solutionPath(filename)).href;

module.exports = {
  readSolution,
  solutionPath,
  solutionUrl
};

