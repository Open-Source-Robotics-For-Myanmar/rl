// Keep the text and viewBox so diagrams remain readable and scale in Markdown.
export default {
  multipass: true,
  plugins: [
    {
      name: 'preset-default',
      params: {
        overrides: {
          removeDesc: false,
          // Mermaid styles select rect/ellipse elements by tag name.
          convertShapeToPath: false,
          convertEllipseToCircle: false,
          // Sequence participants use name attributes in color selectors.
          removeUnknownsAndDefaults: { unknownAttrs: false, keepRoleAttr: true },
        },
      },
    },
    // Preserve inline !important declarations: presentation attributes have
    // lower priority than Mermaid's shared styles and would change colors.
    // Mermaid's debugging attributes are unused by standalone image files.
    { name: 'removeAttrs', params: { attrs: 'data-.*' } },
  ],
};
