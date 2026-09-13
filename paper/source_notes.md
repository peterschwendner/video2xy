# Source evidence and limits

This package describes the implemented revision cc7fe7a82c60362b43f500489821624e65326bf2. Its preserved public baseline is 6a7d1dd8be7622966f6e4d7e5f2dfbdec1001af6. Both experiment scripts were executed against the revision identified in their result JSON. The public remote still pointed to the baseline during this work. The updated code was committed locally and was not pushed.

The baseline module is copied directly from Git. The comparison JSON stores SHA-256 hashes of both module files as read during execution; the code snapshot in this archive preserves those working-file bytes. Git may normalize line endings when checking out the bundle, without changing program content.

The original three-rectangle and raster/clock measurements were rerun and reproduced unchanged. Newly measured results include the resampler compatibility sweep, 25 fps PCM comparison, strict error handling, and stage timings. Old review scripts asserting the presence of now-fixed defects are deliberately excluded from this current reproduction package. The current comparison script explicitly loads the preserved baseline where old behavior is being demonstrated.

The supplied Claude review proposed right insertion for resampling and a counter-only clock correction. Neither proposal was accepted unchanged. Left insertion retains the original boundary rule, while rejecting frame rates above the sample rate avoids a logically incompatible duration/sample-minimum requirement. Infeasible allocation was reachable from the old scan builder and could collapse a rectangle to one repeated point. These findings are now regression tests rather than unresolved current-code claims.

No unverified Claude timings or coverage percentages are used as measured evidence. Hardware playback, camera acquisition, GUI preview, hosted CI, and sustained real-time throughput remain unverified. A simulated audio-start failure is identified as such. The 25 fps test isolates scheduling with one simplified contour; it does not establish invariance for arbitrary static raster extraction.

## Literature checks

- Oscilloscope Music's official description confirms that the same left and
  right channels drive the speakers and the X and Y inputs:
  https://oscilloscopemusic.com/info/about/
- Hansi Raber's OsciStudio page identifies its 2016 provenance and support for
  3D shapes, animations, plugins, and timeline parameters:
  https://asdfg.me/oscistudio/
- Derek Holzer's May 2017 workshop report was read directly. It describes
  multiplexing and separate Z blanking in the Vector Synthesis work:
  https://macumbista.net/wp-content/uploads/2017/05/VECTOR-SYNTHESIS-PURE-DATA-LIBRARY.pdf
- osci-render's repository and v2.2.0 release notes confirm overlapping inputs,
  including PNG, JPEG, and animated GIF. No superiority over that system is claimed:
  https://github.com/jameshball/osci-render
  https://github.com/jameshball/osci-render/releases/tag/v2.2.0
- Canny's original paper was consulted in the Princeton-hosted copy. Correct
  publication details are PAMI-8(6), 679-698, November 1986, rather than the June
  date shown in some secondary records:
  https://www.cs.princeton.edu/courses/archive/fall13/cos429/papers/Canny86.pdf
- Suzuki and Abe's paper metadata and abstract were checked against the publisher
  and Crossref. Crossref malformed the second author as "KeiichiA be"; the
  bibliography uses Keiichi Abe:
  https://doi.org/10.1016/0734-189X(85)90016-7
- Douglas and Peucker's title, authors, volume, issue, pages, and original DOI
  were checked with Crossref and the publisher's later reprint record. The
  reference cites the 1973 original, not the 2011 reprint:
  https://doi.org/10.3138/FM57-6770-U75U-7727
  https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470669488.ch2
- Zuiderveld's chapter details were checked using the official Graphics Gems
  archive and Crossref. The book is Graphics Gems IV; the archive identifies
  Academic Press, 1994, pp. 474-485:
  https://www.realtimerendering.com/resources/GraphicsGems/gems.html
  https://www.realtimerendering.com/resources/GraphicsGems/
- OpenCV 4.13.0 reference pages document the relevant primitives. They are
  documentation references, while the executed verification environment is
  separately identified as OpenCV 5.0.0:
  https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html
  https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html
- Shannon's original publication metadata was checked at IEEE Xplore:
  https://ieeexplore.ieee.org/document/1697831


- NumPy's official searchsorted documentation was consulted for left/right insertion semantics. Its current page identifies NumPy 2.5; it supports the segment-selection explanation, not the experimental equivalence claim:
  https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html

## AI disclosure

Codex assisted with the original project and implemented this revision, wrote and executed tests and experiments, and prepared and rendered the manuscript. Two user-supplied reviews attributed to Claude informed the work. Original model identifiers and prompt history were not supplied and are not invented. The manuscript attributes final responsibility to Peter Schwendner and does not claim that human approval has already occurred.
