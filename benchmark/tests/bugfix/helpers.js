const fs = require('fs');
const path = require('path');

const BUGGY_DIR = path.join(__dirname, '../../fixtures/buggy-components');
const FIXED_DIR = path.join(__dirname, '../../solutions/bugfix');

function getBugfixVariant() {
  return process.env.BUGFIX_VARIANT === 'buggy' ? 'buggy' : 'fixed';
}

function resolveBugfixPath(taskId) {
  const variant = getBugfixVariant();
  const directory = variant === 'buggy' ? BUGGY_DIR : FIXED_DIR;
  const suffix = variant === 'buggy' ? '.buggy.' : '.fixed.';
  const matches = fs
    .readdirSync(directory)
    .filter((fileName) => fileName.startsWith(`${taskId}${suffix}`));

  if (matches.length !== 1) {
    throw new Error(`Expected exactly one ${variant} file for ${taskId}, found ${matches.length}`);
  }

  return path.join(directory, matches[0]);
}

function loadBugfixModule(taskId) {
  const filePath = resolveBugfixPath(taskId);
  delete require.cache[require.resolve(filePath)];
  return require(filePath);
}

function loadBugfixDefaultExport(taskId) {
  const loadedModule = loadBugfixModule(taskId);
  return loadedModule && loadedModule.default ? loadedModule.default : loadedModule;
}

function readBugfixSource(taskId) {
  return fs.readFileSync(resolveBugfixPath(taskId), 'utf8');
}

module.exports = {
  getBugfixVariant,
  loadBugfixDefaultExport,
  loadBugfixModule,
  readBugfixSource
};
