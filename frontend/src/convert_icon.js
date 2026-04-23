const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');
const path = require('path');
const { imagesToIco } = require('png-to-ico');

async function convertSvgToIco(svgPath, outputDir) {
    const svgContent = fs.readFileSync(svgPath, 'utf8');

    const sizes = [16, 32, 48, 64, 128, 256];
    const pngPaths = [];
    
    // Generate PNGs for each size
    for (const size of sizes) {
        const resvg = new Resvg(svgContent, {
            fitTo: {
                mode: 'width',
                value: size
            }
        });

        const pngData = resvg.render();
        const pngBuffer = pngData.asPng();

        const pngPath = path.join(outputDir, `icon_${size}.png`);
        fs.writeFileSync(pngPath, pngBuffer);
        pngPaths.push(pngPath);
        console.log(`Created: ${pngPath}`);
    }

    // 256x256 PNG for electron-builder
    const mainPngPath = path.join(outputDir, 'icon.png');
    fs.copyFileSync(path.join(outputDir, 'icon_256.png'), mainPngPath);
    console.log(`Created: ${mainPngPath}`);

    // Create ICO file from PNGs
    const icoPath = path.join(outputDir, 'icon.ico');
    const pngBuffers = pngPaths.map(p => fs.readFileSync(p));
    const icoBuffer = await imagesToIco(pngBuffers);
    fs.writeFileSync(icoPath, icoBuffer);
    console.log(`Created: ${icoPath}`);

    console.log('\nDone! Icon files generated.');
}

const svgPath = process.argv[2] || 'icons/icon_arc_wing4b.svg';
const iconsDir = path.join(__dirname, 'icons');
convertSvgToIco(svgPath, iconsDir);
