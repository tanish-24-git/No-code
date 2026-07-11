/**
 * Side-effect barrel: importing this module registers every tool.
 * Lives OUTSIDE index.ts so the registry Map is initialized before any
 * registerTool() call (ESM imports are hoisted).
 */
import './run-terminal';
import './fs';
import './ask-user';
import './propose-plan';
