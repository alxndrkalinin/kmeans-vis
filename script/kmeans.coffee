console.log 'K-means clustering'

DataManager = ( ->

  _url = null
  _easyUrl =
    data: 'data/easyA.csv'
    labels: 'data/easyZ.csv'
  _notEasyUrl =
    data: 'data/noteasyA.csv'
    labels: 'data/noteasyZ.csv'

  _data = []
  _labels = []

  _loadData = (dataSet='easy') ->

    switch dataSet
      when 'easy' then _url = _easyUrl
      when 'not-easy' then _url = _notEasyUrl

    Papa.parse _url.data,
      download: true,
      complete: (dataResults) ->
        if dataResults and dataResults.data?.length > 0
          _data = dataResults.data
          _data.pop()
          Papa.parse _url.labels,
            download: true,
            complete: (labelsResults) ->
              if labelsResults and labelsResults.data?.length > 0
                _labels = labelsResults.data
                _labels.pop()
                _sendDataToCore
                  data: _data
                  labels: _labels

  _sendDataToCore = (data) ->
    console.log 'DataManager: data loaded'
    console.log data
    Core.fireEvent
      msg : 'data_loaded'
      data: data

  loadData: _loadData
)()

KMeans = ( ->

  _getUniqueLabels = (labels) ->
    labels.filter (x, i, a) -> i is a.indexOf x

  _distance = (v1, v2, type='euclidean', s=[]) ->

    euclidean = (v1, v2) ->
      total = 0
      for i in [0..v1.length - 1]
        total += Math.pow(v2[i] - v1[i], 2)
      Math.sqrt(total)

    manhattan = (v1, v2) ->
      total = 0
      for i in [0..v1.length - 1]
        total += Math.abs(v2[i] - v1[i]);
      total

    max = (v1, v2) ->
      max = 0
      for i in [0..v1.length - 1]
        max = Math.max(max , Math.abs(v2[i] - v1[i]));
      max

    mahalanobis = (v1, v2, s) ->

      l = v1.length
      invCov = s

      diff = (v1[k] - v2[k] for k in [0..l - 1])
      total = 0
      for row, i in invCov
        for el, j in row
          total += invCov[i][j] * Math.pow(diff[i], 2 - i - j) * Math.pow(diff[j], i + j)
      Math.sqrt(total)

    if _arrayEqual v1, v2
      return 0
    else
      switch type.toLowerCase()
        when 'manhattan' then return manhattan v1, v2
        when 'max' then return max v1, v2
        when 'mahalanobis' then return mahalanobis v1, v2, s
        else return euclidean v1, v2

  _arrayEqual = (x, y) ->
    a = x.slice().sort()
    b = y.slice().sort()
    a.length is b.length and a.every (elem, i) -> elem is b[i]

  _matrixMultiply = (a, b) ->
    c = (0 for d1 in a.length for d2 in d3.transpose(b))
    for row, i in a
      for col, j in d3.transpose(b)
        c[i][j] = (row[k] * col[k] for el, k in row).reduce (t, s) -> t + s
    c

  _initCentroids = (data, k) ->
    nRows = data.length
    centroids = []
    for ctr in [0..k - 1]
      ctrIdx = Math.floor(Math.random() * nRows)
      if centroids.length and ctrIdx is not centroids[ctr - 1].idx
        ctrIdx = Math.floor(Math.random() * nRows)
      centroids.push
        val: data[ctrIdx]
        idx: ctrIdx
    centroids

  _initLabels = (l, k) ->
    labels = []
    labels.push Math.floor(Math.random() * k) for i in [0..l]
    labels

  _updateMeans = (data, centroids, labels) ->
    ctrData = ([] for ctr in centroids)
    for row, rowIdx in data
      ctrData[labels[rowIdx]].push row

    means = []
    for ctr, ctrIdx in ctrData
      colSums = (0 for col in data[0])
      for row in ctr
        for el, elIdx in row
          colSums[elIdx] += el
      colMeans = colSums.map (x) -> x / ctr.length
      means.push colMeans

    centroids = []
    for mean in means
      distances = (_distance(row, mean, 'euclidean') for row in data)
      ctrIdx = distances.indexOf(Math.min.apply @, distances)
      # trying not to assign the same point
      if ctrIdx not in centroids.map((x) -> x.idx)
        centroids.push
          val: data[ctrIdx]
          idx: ctrIdx
      else
        distances = distances.splice(ctrIdx)
        ctrIdx = distances.indexOf(Math.min.apply @, distances)
        centroids.push
          val: data[ctrIdx]
          idx: ctrIdx
    centroids

  _updatePrecisionMatrix = (data, ctrIdx, labels) ->
    matrix = []
    for i in [0..data.length]
      if labels[i] is ctrIdx
        matrix.push data[i].slice()
    #      (matrix.push(row) for row, i in data when labels[i] is ctrIdx)
    n = matrix.length

    matrixT = d3.transpose matrix
    l = matrixT.length
    means = (col.reduce((t, s) -> t + s) / n for col in matrixT)

    for row, i in matrix
      for col, j in row
        matrix[i][j] = col - means[j]
    matrixT = d3.transpose(matrix)

    cov = (0 for e1 in [0..l - 1] for e2 in [0..l - 1])
    cov = _matrixMultiply matrixT, matrix
    cov = cov.map((row) -> row.map((el) -> el / (n - 1)))

    # calculate pseudo-inverse covariance matrix
    tCov = new jsfeat.matrix_t l, l, jsfeat.F32_t | jsfeat.C1_t
    covData = []
    (covData.push(e) for e in row for row in cov)
    tCov.data = covData
    tCovInv = new jsfeat.matrix_t l, l, jsfeat.F32_t | jsfeat.C1_t
    jsfeat.linalg.svd_invert tCovInv, tCov

    invCov = (0 for e1 in [0..l - 1] for e2 in [0..l - 1])
    for row, i in invCov
      for col, j in row
        invCov[i][j] = tCovInv.data[2 * i + j]

    invCov

  _assignSamples = (data, centroids, distanceType) ->
    labels = []
    for row in data
      distances = (_distance(row, ctr.val, distanceType) for ctr in centroids)
      labels.push distances.indexOf(Math.min.apply @, distances)
    labels

  _runKMeans = (data, trueLabels, k, maxIter, centroids, distanceType, uniqueLabels, computeAcc) ->

    evaluateAccuracy = (labels, trueLabels, uniqueLabels) ->
      accs = [0]
      for k in uniqueLabels
        kLabelIdxs = (i for x, i in trueLabels when x is k)
        kLabels = (x for x, i in labels when i in kLabelIdxs)
        kTrueLabels = (x for x in trueLabels when x is k)
        accK = kLabels.map((x, idx) -> x - kTrueLabels[idx]).reduce (r, s) -> r + s
        accs.push Math.abs(accK)
      acc = (trueLabels.length - accs.reduce((r, s) -> r + s)) / trueLabels.length
      acc = if acc < 0.5 then 1 - acc else acc

    step = (data, centroids, trueLabels) ->
      maxIter--
      console.log 'Iteration: ' + maxIter
      console.log 'Centroids: '
      console.table centroids

      labels = _assignSamples data, centroids, distanceType
      means = _updateMeans data, centroids, labels

      console.log 'New means: '
      console.table means
      if not _arrayEqual means.map((x) -> x.idx), centroids.map((x) -> x.idx)
        centroids = means
        Controller.redraw(data, means.map((x) -> x.val), labels)
      else
        maxIter = 0

      centroids: centroids
      labels: labels

    run = () ->
      # main loop
      if maxIter
        res = step data, centroids, trueLabels
        centroids = res.centroids
        labels = res.labels
      else
        clearInterval interval
        console.log 'K-Means done.'
        if computeAcc
          labels = _assignSamples data, centroids, distanceType
          acc = evaluateAccuracy labels, trueLabels, uniqueLabels
          console.log 'Accuracy: ' + acc * 100 + '%'
        else
          acc = ''
        Core.fireEvent
          msg: 'kmeans_done'
          data: acc * 100

    runMahalanobis = () ->
      # main loop
      if maxIter
        maxIter--
        console.log 'Iteration: ' + maxIter
        console.log 'Centroids: '
        console.table centroids

        means = centroids.slice()

        for row, i in data

          ctrDistances = (_distance(row, ctr.val, distanceType, covMats[j]) for ctr, j in centroids)
          ctrIdx = ctrDistances.indexOf(Math.min.apply @, ctrDistances)

          if ctrIdx isnt lbls[i]
            lbls[i] = ctrIdx
            centroids = _updateMeans data, centroids, lbls
            Controller.redraw(data, centroids.map((x) -> x.val), lbls)
            for ctr, j in centroids
              covMats[j] = _updatePrecisionMatrix(data, j, lbls)

        if _arrayEqual(means.map((x) -> x.idx), centroids.map((x) -> x.idx))
          maxIter = 0

      else
        clearInterval interval
        console.log 'K-Means done.'
#        labels = _assignSamples data, centroids, distanceType
        if computeAcc
          acc = evaluateAccuracy lbls, trueLabels, uniqueLabels
          console.log 'Accuracy: ' + acc * 100 + '%'
        else
          acc = ''
        Core.fireEvent
          msg: 'kmeans_done'
          data: acc * 100

    if distanceType is 'mahalanobis'
      labels = _assignSamples data, centroids, 'euclidean'
      centroids = _updateMeans data, centroids, labels
      covMats = []
      lbls = labels.slice()
      for ctr, ctrIdx in centroids
        covMats.push _updatePrecisionMatrix(data, ctrIdx, labels)
      interval = setInterval runMahalanobis, 1000
    else
      interval = setInterval run, 1000

  init = (obj, k, maxIter=20) ->
    data = obj.data
    data = (row.map(Number) for row in data)
    labels = obj.labels
    labels = labels.map (x) -> Number(x[0])
    computeAcc = on

    distanceType = Controller.getDistanceType()
    uniqueLabels = _getUniqueLabels(labels)

    k = Number k
    console.log 'K: ' + k

    if k isnt 2
      computeAcc = off

    Controller.drawDataPoints data

    initMethod = Controller.getInitMethod()
    if initMethod is 'forgy'
      centroids = _initCentroids data, k
      initLabels = _assignSamples data, centroids, 'euclidean'
    else
      initLabels = _initLabels data.length - 1, k
      centroids = _updateMeans data, uniqueLabels, initLabels

    Controller.redraw data, centroids.map((x) -> x.val), initLabels

    console.log 'Starting K-Means'
    _runKMeans data, labels, k, maxIter, centroids, distanceType, uniqueLabels, computeAcc

  run: init
)()

Controller = ( ->

  _graph = null
  _xScale = null
  _yScale = null
  _color = null
  _meanLayer = null

  _getDistanceType = () ->
    $('.distInput input:radio:checked').val()

  _getDataset = () ->
    $('.dataInput input:radio:checked').val()

  _getInitMethod = () ->
    $('.initInput input:radio:checked').val()

  _getKValue = () ->
    $('.kInput select').val()

  _setStartButtonListener = (func) ->
    $('.runKMeans').click () ->
      event.preventDefault()
      _clearChart()
      func()

  _drawDataPoints = (dataPoints) ->
    pointDots = _graph.selectAll('.pointDots').data(dataPoints)
    pointDots.enter().append('circle').attr('class','pointDots')
      .attr('r', 3)
      .attr('cx', (d) -> _xScale(d[0]))
      .attr('cy', (d) -> _yScale(d[1]))

  _clearChart = () ->
    _graph.selectAll('.pointDots').remove()
    _graph.selectAll('g > g > *').remove()

  _redraw = (dataPoints, means, assignments) ->
    assignmentLines = _meanLayer.selectAll('.assignmentLines').data(assignments)
    assignmentLines.enter().append('line').attr('class','assignmentLines')
      .attr('x1', (d, i) -> _xScale(dataPoints[i][0]))
      .attr('y1', (d, i) -> _yScale(dataPoints[i][1]))
      .attr('x2', (d, i) -> _xScale(means[d][0]))
      .attr('y2', (d, i) -> _yScale(means[d][1]))
      .attr('stroke', (d) -> _color(d))

    assignmentLines.transition().duration(500)
      .attr('x2', (d, i) -> _xScale(means[d][0]))
      .attr('y2', (d, i) -> _yScale(means[d][1]))
      .attr('stroke', (d) -> _color(d))

    meanDots = _meanLayer.selectAll('.meanDots').data(means)
    meanDots.enter().append('circle').attr('class','meanDots')
      .attr('r', 5)
      .attr('stroke', (d, i) -> _color(i))
      .attr('stroke-width', 3)
      .attr('fill', 'white')
      .attr('cx', (d) -> _xScale(d[0]))
      .attr('cy', (d) -> _yScale(d[1]))

    meanDots.transition().duration(500)
      .attr('cx', (d) -> _xScale(d[0]))
      .attr('cy', (d) -> _yScale(d[1]))
    meanDots.exit().remove()

  _publishResult = (data) ->
    $('.kMeansAcc').text data + '%'

  _initChart = () ->
    svg = d3.select('#chart').append('svg').attr('width', 1200).attr('height', 700)
    _graph = svg.append('g').attr('transform', 'translate(350,350)')
    _meanLayer = _graph.append('g')
    _xScale = d3.scale.linear().domain([0,10]).range([0,300])
    _yScale = d3.scale.linear().domain([0,10]).range([0,300])
    _color = d3.scale.category10()

  getDistanceType: _getDistanceType
  getDataset: _getDataset
  getInitMethod: _getInitMethod
  getKValue: _getKValue
  setStartButtonListener: _setStartButtonListener
  drawDataPoints: _drawDataPoints
  publishResult: _publishResult
  initChart: _initChart
  redraw: _redraw
)()

Core = ( ->

  _start = () ->
    Controller.setStartButtonListener _getData
    Controller.initChart()
    _getData()

  _getData = () ->
    dataset = Controller.getDataset()
    DataManager.loadData(dataset)
    console.log 'Dataset selected: ' + dataset

  _startKMeans = (data) ->
    k = Controller.getKValue()
    KMeans.run data, k

  _api = (ev) ->

    if ev? & ev.msg? & ev.data?

      switch ev.msg
        when 'data_loaded' then _startKMeans ev.data
        when 'kmeans_done' then Controller.publishResult ev.data
        else console.log 'Unknown message: ' + ev.msg

    else
      return false

  _init = () ->
    console.log 'Core init'
    _start()


  init: _init
  fireEvent: _api
)()

return start: Core.init()