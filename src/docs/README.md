# PyGent Factory Documentation

This directory contains the complete documentation for PyGent Factory, built with VitePress for a modern, fast, and beautiful documentation experience.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

### Build for Production

```bash
# Build static site
npm run build

# Preview production build
npm run preview
```

## 📁 Documentation Structure

```
docs/
├── .vitepress/           # VitePress configuration
│   ├── config.ts         # Main configuration
│   └── theme/            # Custom theme
│       ├── index.ts      # Theme entry point
│       ├── custom.css    # Custom styles
│       └── components/   # Vue components
├── getting-started/      # Getting started guides
├── concepts/            # Core concepts
├── guides/              # User guides
├── api/                 # API reference
├── examples/            # Examples and tutorials
├── advanced/            # Advanced topics
└── index.md             # Homepage
```

## 🎨 Features

### Modern Documentation Framework
- **VitePress**: Lightning-fast static site generation
- **Vue 3**: Interactive components and examples
- **TypeScript**: Type-safe configuration and components
- **Responsive Design**: Mobile-first, accessible design

### Interactive Components
- **CodeExample**: Tabbed code examples with syntax highlighting
- **ArchitectureDiagram**: Mermaid diagrams for system visualization
- **InteractiveDemo**: Live API testing and WebSocket demos
- **FeatureCard**: Beautiful feature showcases

### Advanced Features
- **Search**: Local search with instant results
- **Dark Mode**: Automatic dark/light theme switching
- **Navigation**: Intuitive sidebar and navigation
- **SEO Optimized**: Meta tags, OpenGraph, and structured data

## 🔧 Customization

### Adding New Pages

1. Create a new Markdown file in the appropriate directory
2. Add frontmatter with title and description
3. Update the sidebar configuration in `.vitepress/config.ts`

Example:
```markdown
---
title: "New Feature Guide"
description: "Learn how to use the new feature"
---

# New Feature Guide

Content goes here...
```

### Custom Components

Create Vue components in `.vitepress/theme/components/` and register them in `.vitepress/theme/index.ts`:

```typescript
// Register component
app.component('MyComponent', MyComponent)
```

Use in Markdown:
```markdown
<MyComponent prop="value" />
```

### Styling

Customize styles in `.vitepress/theme/custom.css`:

```css
:root {
  --vp-c-brand-1: #your-color;
}
```

## 📊 Mermaid Diagrams

The documentation supports Mermaid diagrams for architecture visualization:

```markdown
<ArchitectureDiagram
  title="System Overview"
  type="mermaid"
  content="graph TB
    A[Component A] --> B[Component B]
    B --> C[Component C]"
  description="This diagram shows the system flow."
/>
```

## 🧪 Interactive Examples

Add interactive API demos:

```markdown
<InteractiveDemo
  title="Try the API"
  description="Test PyGent Factory endpoints"
  type="api"
  endpoint="/api/v1/agents"
/>
```

## 📝 Writing Guidelines

### Content Structure
- Use clear, descriptive headings
- Include code examples for all concepts
- Add diagrams for complex architectures
- Provide interactive demos where possible

### Style Guide
- Write for beginners but include advanced details
- Use active voice and clear language
- Include practical examples and use cases
- Add troubleshooting sections

### Code Examples
- Use the `CodeExample` component for multi-language examples
- Include complete, runnable code
- Add descriptions explaining the code
- Show both basic and advanced usage

## 🚀 Deployment

### Cloudflare Pages

1. Connect your GitHub repository to Cloudflare Pages
2. Set build command: `npm run build`
3. Set output directory: `dist`
4. Deploy automatically on push

### Netlify

1. Connect your GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Deploy automatically on push

### GitHub Pages

1. Enable GitHub Pages in repository settings
2. Use GitHub Actions for automated deployment:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm install
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: dist
```

### Custom Domain

Add a `CNAME` file to the `public` directory:

```
docs.pygent.ai
```

## 🔍 SEO Optimization

The documentation includes comprehensive SEO optimization:

- **Meta Tags**: Title, description, keywords
- **OpenGraph**: Social media sharing
- **Structured Data**: JSON-LD for search engines
- **Sitemap**: Automatic sitemap generation
- **Performance**: Optimized for Core Web Vitals

## 📈 Analytics

Add analytics tracking in `.vitepress/config.ts`:

```typescript
export default defineConfig({
  head: [
    // Google Analytics
    ['script', { async: true, src: 'https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID' }],
    ['script', {}, `
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'GA_MEASUREMENT_ID');
    `]
  ]
})
```

## 🤝 Contributing

### Documentation Guidelines

1. **Accuracy**: Ensure all code examples work
2. **Completeness**: Cover all features and use cases
3. **Clarity**: Write for your target audience
4. **Consistency**: Follow the established style guide

### Review Process

1. Create a feature branch
2. Make your changes
3. Test locally with `npm run dev`
4. Submit a pull request
5. Address review feedback

### Content Review Checklist

- [ ] All code examples are tested and working
- [ ] Diagrams accurately represent the system
- [ ] Links are working and point to correct pages
- [ ] Content is accessible and follows guidelines
- [ ] SEO metadata is complete and accurate

## 🛠️ Troubleshooting

### Common Issues

**Build Errors**
- Check Node.js version (18+ required)
- Clear node_modules and reinstall dependencies
- Verify all imports and component registrations

**Styling Issues**
- Check CSS custom properties
- Verify component styles are scoped correctly
- Test in both light and dark modes

**Navigation Problems**
- Verify sidebar configuration in config.ts
- Check file paths and naming conventions
- Ensure all referenced files exist

### Getting Help

- Check the VitePress documentation
- Review existing issues in the repository
- Ask questions in the community discussions

## 📚 Resources

- [VitePress Documentation](https://vitepress.dev/)
- [Vue 3 Documentation](https://vuejs.org/)
- [Mermaid Documentation](https://mermaid.js.org/)
- [Markdown Guide](https://www.markdownguide.org/)

---

This documentation is built with ❤️ using VitePress and modern web technologies to provide the best possible developer experience.